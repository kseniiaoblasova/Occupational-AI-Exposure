import pandas as pd
import numpy as np
import anthropic
import requests
from OnetWebService import OnetWebService
import os
from dotenv import load_dotenv

load_dotenv()

_client = None
_onet = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def get_onet():
    global _onet
    if _onet is None:
        _onet = OnetWebService(os.getenv("ONET_API_KEY"))
    return _onet


# ---- Feature column order (must match training data) ----

FEATURE_COLUMNS = [
    "isBright", "isGreen", "JobZone", "MedianSalary",
    "pct_computer", "pct_physical", "pct_communication",
    "pct_analyze", "pct_manage", "pct_creative", "pct_textnative"
]


# ---- System prompts ----

KEYWORD_SYSTEM_PROMPT = (
    "You are an occupational classification specialist. "
    "Your job is to take a user-provided job title (and optionally a job description) "
    "and produce 5 search terms that will find the best matching occupations "
    "in the O*NET database.\n\n"
    "IMPORTANT — Input validation:\n"
    "If the job title is clearly nonsensical, gibberish, or not a real occupation "
    "(e.g. random characters like 'kk', 'sdjksjd', 'asdfgh', phrases like 'don't know', "
    "'no idea', 'idk', 'yehs', or other keyboard mashing), output ONLY the single word:\n"
    "INVALID_JOB_TITLE\n"
    "Nothing else. No explanation.\n\n"
    "Important context about O*NET:\n"
    "- O*NET contains approximately 1,016 formally titled U.S. occupations "
    "(e.g. 'Software Developers', 'Registered Nurses', 'Market Research Analysts').\n"
    "- It uses formal, standardized language — not casual titles like 'coder' or 'number cruncher'.\n"
    "- Many modern job titles (Prompt Engineer, AI Ethicist, Growth Hacker, Influencer) "
    "do NOT exist in O*NET. For these, you must suggest the closest traditional occupations "
    "that share the core tasks.\n\n"
    "Rules (for valid inputs only):\n"
    "1. Always include the original job title (or a lightly cleaned version) as the first keyword.\n"
    "2. The remaining 4 keywords should be plausible O*NET occupation titles or short phrases "
    "that capture different aspects of the job.\n"
    "3. Prioritize breadth — cover different facets of the role rather than synonyms.\n"
    "4. Output ONLY the 5 keywords separated by commas. No numbering, no explanation, nothing else.\n\n"
    "Example:\n"
    "Input: 'Prompt Engineer'\n"
    "Output: prompt engineer, technical writer, software quality assurance analyst, "
    "artificial intelligence specialist, training and development specialist"
)

OCCUPATION_SELECTION_SYSTEM_PROMPT = (
    "You are an occupational classification specialist. "
    "You will receive a job title, an optional job description, "
    "and a list of candidate O*NET occupations found by keyword search.\n\n"
    "Your job: pick the ONE occupation from the list whose tasks would best "
    "represent the day-to-day work of the given job title.\n\n"
    "Rules:\n"
    "- Ignore 'catch-all' categories like 'All Other' occupations — these are vague and unhelpful.\n"
    "- If the job is very new (e.g. Prompt Engineer, AI Ethicist), pick the traditional occupation "
    "whose core tasks overlap most with what this person actually does.\n"
    "- Output ONLY the SOC code (e.g. '15-1252.00'). Nothing else.\n"
)
# ======================================================================
# O*NET API
# ======================================================================


def _generate_keywords(job_title, job_description):
    """use claude to generate onet search keywords from job title"""
    user_content = f"Job title: {job_title}"
    if job_description:
        user_content += f"\nJob description: {job_description}"

    print(f"\n[Claude] generate_keywords ← {repr(user_content)}")
    response = get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=KEYWORD_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()
    print(f"[Claude] generate_keywords → {repr(raw)}")
    if raw.upper() == "INVALID_JOB_TITLE":
        return None  # signals nonsense input
    return [k.strip() for k in raw.split(",") if k.strip()]


def _search_onet(keywords):
    """search onet for each keyword, deduplicate by soc code"""
    candidates = {}

    for keyword in keywords:
        print(f"[O*NET] search ← {repr(keyword)}")
        results = get_onet().call(
            "online/search",
            ("keyword", keyword),
            ("end", 5),
        )

        if "occupation" not in results:
            print(f"[O*NET] search → no results")
            continue

        hits = results["occupation"]
        print(
            f"[O*NET] search → {len(hits)} result(s): {[o['code'] for o in hits]}")
        for occ in hits:
            code = occ["code"]
            if code not in candidates:
                candidates[code] = {
                    "code": code,
                    "title": occ["title"],
                }

    print(f"[O*NET] total unique candidates: {len(candidates)}")
    return candidates


def _select_best_occupation(job_title, job_description, candidates):
    """claude picks the most relevant occupation from candidates"""
    candidate_list = "\n".join(
        f"- {c['code']}: {c['title']}" for c in candidates.values()
    )

    user_content = f"Job title: {job_title}\n"
    if job_description:
        user_content += f"Job description: {job_description}\n"
    user_content += f"\nCandidate occupations:\n{candidate_list}"

    print(
        f"\n[Claude] select_occupation ← job_title={repr(job_title)}, {len(candidates)} candidates")
    response = get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        system=OCCUPATION_SELECTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    soc = response.content[0].text.strip()
    title = candidates.get(soc, {}).get("title", "?")
    print(f"[Claude] select_occupation → {soc} ({title})")
    return soc


def _fetch_tasks(soc_code):
    """fetch all task statements (request up to 50 to skip pagination)"""
    print(f"[O*NET] fetch_tasks ← {soc_code}")
    result = get_onet().call(
        f"online/occupations/{soc_code}/details/tasks",
        ("end", 50),
    )

    if "error" in result:
        print(f"[O*NET] fetch_tasks → error: {result['error']}")
        return []

    tasks = []
    for task in result.get("task", []):
        statement = task.get("title", "")
        if statement:
            tasks.append(statement.strip().lower())

    print(f"[O*NET] fetch_tasks → {len(tasks)} tasks")
    return tasks


def _fetch_job_zone(soc_code):
    """pull job zone (1-5) for an occupation"""
    print(f"[O*NET] fetch_job_zone ← {soc_code}")
    result = get_onet().call(f"online/occupations/{soc_code}/details/job_zone")

    if "error" in result:
        print(f"[O*NET] fetch_job_zone → error: {result['error']}")
        return None

    value = result.get("job_zone", {}).get("value", None)
    print(f"[O*NET] fetch_job_zone → {value}")
    return value


def _fetch_occupation_flags(soc_code):
    """pull isBright and isGreen from onet occupation tags"""
    print(f"[O*NET] fetch_flags ← {soc_code}")
    result = get_onet().call(f"online/occupations/{soc_code}")

    if "error" in result:
        print(f"[O*NET] fetch_flags → error: {result['error']}")
        return {"isBright": None, "isGreen": None}

    tags = result.get("tags", {})
    flags = {
        "isBright": int(tags.get("bright_outlook", False)),
        "isGreen": int(tags.get("green", False))
    }
    print(
        f"[O*NET] fetch_flags → isBright={flags['isBright']}, isGreen={flags['isGreen']}")
    return flags


# ======================================================================
# BLS OEWS API
# ======================================================================

def _fetch_bls_median_salary(soc_code):
    """
    fetch median annual salary from BLS OEWS public data API v2.
    series ID: OEUN + 17-char area code (national) + 6-digit soc + 13 (median annual)
    register for free api key at https://data.bls.gov/registrationEngine/
    set BLS_API_KEY in .env for 500 requests/day (vs 25 without).
    """
    # "15-1252.00" -> "151252"
    clean_soc = soc_code.split('.')[0].replace("-", "")
    if len(clean_soc) < 6:
        clean_soc = clean_soc.ljust(6, "0")
    clean_soc = clean_soc[:6]

    series_id = f"OEUN0000000000000{clean_soc}13"

    print(f"[BLS] fetch_salary ← series_id={series_id}")
    try:
        payload = {
            "seriesid": [series_id],
            "startyear": "2023",
            "endyear": "2024",
        }

        bls_key = os.getenv("BLS_API_KEY")
        if bls_key:
            payload["registrationkey"] = bls_key

        resp = requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json=payload, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            print(f"[BLS] fetch_salary → failed (status={data.get('status')})")
            return None

        series = data.get("Results", {}).get("series", [])
        if not series or not series[0].get("data"):
            print(f"[BLS] fetch_salary → no data returned")
            return None

        value = series[0]["data"][0].get("value")
        result = float(value) if value else None
        print(
            f"[BLS] fetch_salary → ${result:,.0f}" if result else "[BLS] fetch_salary → None")
        return result

    except Exception as e:
        print(f"[BLS] fetch_salary → exception: {e}")
        return None


# ======================================================================
# Feature engineering
# ======================================================================

def _keyword_features(tasks):
    """apply keyword patterns to task list, return df with 7 binary task flags"""
    task_features = pd.DataFrame({"task_name": tasks})

    task_features['has_computer'] = task_features['task_name'].str.contains(
        'compute|computer|software|database|digital|program(?:ming)?|web|network(?:ing)?|'
        'server|cloud|spreadsheet|online|internet|website|algorithm|cyber|code|coding|'
        'automat(?:e|ed|ion)|electronic.*system|information.*system|data.*system',
        case=False).astype(int)

    task_features['is_physical'] = task_features['task_name'].str.contains(
        'car|seal|push|open|attend|participate|vacuum|visit|wash|glue|stuch|attach|wax|watch|'
        'water|wear|weigh|wet|whiten|administer|wind|witness|add|wipe|activate|store|distribute|'
        'hospital|wrap|accompany|technician|repair|assemble|install|load|unload|lift|driv(?:e|ing)|'
        'clean(?:ing)?|transport|weld|drill|grind|shovel|plow|harvest|plant(?:ing)|irrigat|'
        'slaughter|butcher|stack|construct|demolish|excavat|pour|spray|mount|fasten|bolt|wire|'
        'mail|solder|patrol|guard|restrain|rescue|extinguish|cook|bake|grill|fry|chop|sew|forge|'
        'hoist|crane|tow|pave|plumb|drywall|cement|brick|landscap|prune|fertiliz|fumigat|deliver|'
        'carry|mow|remove|adjust|measur|mix|cut|attach|sort|run|jump|sport|polish|paint|haul|'
        'operate|machine|hand.*tool|physical|guest(?:hands.on|face.to.face|in.person|bedside|'
        'chairside|tableside)|(?:administer|inject|infuse|insert|suture|incision|intubat).*'
        '(?:patient|medic|drug|treatment|anesthes|iv\\b|blood|fluid|needle|catheter)|'
        '(?:physical.*exam|examine.*patient|examine.*client)|'
        '(?:massage|manipulat.*(?:joint|spine|tissue|muscle|body))|'
        '(?:therapy|therapeutic).*(?:session|exercise|technique|intervention|treatment)|'
        '(?:restrain|immobili|splint|bandage|dress.*wound)|'
        '(?:escort|accompany).*(?:patient|client|resident|guest|visitor|prisoner)|'
        'transport.*(?:patient|client|resident)|'
        '(?:bathe|groom|toileting|hygiene).*(?:patient|client|resident)|'
        '(?:vital.sign|blood.pressure|pulse|heart.rate)(?!.*(?:data|record|monitor.*system))|'
        '(?:arrest|apprehend|detain|handcuff|frisk)|(?:perform.*surgery|surgical.*procedure)|'
        '(?:serve.*food|wait.*table|seat.*guest|bus.*table)|'
        '(?:fit|measure|alter).*(?:prosthe|orthotic|hearing.aid|eyeglass|lens|brace)|'
        '(?:child|children|infant|toddler).*(?:care|supervis|watch|feed|bathe)',
        case=False
    ).astype(int)

    task_features['must_communicate'] = task_features['task_name'].str.contains(
        'nominate|debate|propose|represent|promote|encourage|support|language|tone|warn|accept|'
        'team|child|adult|adolescent|other|director|member|crew|staff|arrange|inform|feedback|'
        'train|hire|recruit|supervise|communicat|present(?:ation)|negotiat|counsel|advise|'
        'advis(?:e|ing|ory)|teach(?:ing)?|interview|consult|collaborat|coordinat|mediat|'
        'facilitat|mentor|coach(?:ing)?|instruct|lectur|tutor|persuad|advocat|arbitrat|liais|'
        'confer|discuss|explain|translat|greet|testif|preach|officiat|client(?:s)?|customer(?:s)?|'
        'patient(?:s)?|student(?:s)?|notify|recommend|correspond|'
        'respond.*(?:inquir|request|complaint|question)',
        case=False
    ).astype(int)

    task_features['must_analyze'] = task_features['task_name'].str.contains(
        'goasl|deadline|knowledge|identify|market|strateg|promote|order|decision|resolve|adhere|'
        'adapt|idea|arrange|write|diagnos|analyz|research(?:ing)?|investigat|evaluat|assess|'
        'calculat|estimat|forecast|predict|statistic|audit(?:ing)?|diagnos|synthesiz|verify|'
        'validat|classif|apprais|benchmark|hypothes|feasib|'
        '(?:conduct|perform|design).*(?:stud|experiment|survey|research|test|inspection)|'
        'review|examine|inspect|compile|determin|monitor',
        case=False
    ).astype(int)

    task_features['must_manage'] = task_features['task_name'].str.contains(
        'direct|alert|manage|manag(?:ing|ement)|supervis|oversee|oversight|delegat|authoriz|'
        'approv(?:e|al)|budget|allocat|procurement|hire|hiring|recruit|schedul(?:e|ing)|assign|'
        'organiz|implement|establish|direct.*(?:work|activit|staff|operation)|'
        'plan.*(?:activit|project|program|operation|strateg)',
        case=False
    ).astype(int)

    task_features['is_creative'] = task_features['task_name'].str.contains(
        'marketing|warn|artwork|artistic|aesthetic|creative|illustrat|sculpt|choreograph|'
        'storyboard|lyric|photograph(?:y|ing|er|ic)|cinematograph|decorat(?:e|ing|ion|ive)|'
        'compose.*(?:music|song|melody|score|vocal)|(?:arrange|transpose|transcribe).*music|'
        'play\\b.*(?:instrument|piano|guitar|violin|drum|trumpet|flute)|'
        '(?:perform|rehearse).*(?:music|dance|song|concert|recital|comedy|drama|skit)|'
        'sketch.*(?:design|drawing|art|set|idea|concept)|'
        '(?:design|create|develop).*(?:artwork|illustration|animation|graphic|logo|visual|'
        'display|layout)|write.*(?:script|story|fiction|poem|poetry|lyric|novel|press.release|'
        'advertisement|song|article|blog|editorial|commentary|column)|graphic.design|'
        'interior.design|fashion.design|costume.design|floral.design|set.design|web.design|'
        'landscape.design|(?:direct|produce|edit).*(?:film|video|movie|broadcast|production|'
        'documentary|animation)|(?:create|develop|design).*(?:recipe|menu)|photograph',
        case=False
    ).astype(int)

    task_features['is_text_native'] = task_features['task_name'].str.contains(
        'write.*(?:code|report|document|email|letter|memo|proposal|brief|summary|article|script|'
        'manual|specification|plan|policy|procedure|grant|contract|review|description|standard|'
        'guideline|instruction|recommendation)|'
        'draft.*(?:report|document|letter|memo|proposal|brief|correspondence|contract|'
        'legislation|policy|plan|budget)|'
        'prepare.*(?:report|document|correspondence|brief|presentation|statement|tax|return|'
        'invoice|budget|proposal|spreadsheet|manuscript)|'
        'compose.*(?:email|letter|memo|correspondence|message)|'
        'transcri(?:be|ption)|data.entry|enter.*data|key.*data|input.*data|'
        'edit.*(?:text|document|manuscript|copy|report|content|draft)|proofread|'
        'program(?:ming)?\\b.*(?:software|computer|application|system|script|code|language)|'
        'develop.*(?:software|code|application|website|database|script|algorithm)|debug|'
        'compil(?:e|ing).*(?:data|report|record|information|statistic|document)|spreadsheet|'
        'bookkeep|ledger|translat(?:e|ing|ion).*(?:document|text|material|content)|word.process',
        case=False
    ).astype(int)

    return task_features


def _aggregate_task_features(task_features_df):
    """collapse task-level binary flags into occupation-level pct features (mean)"""
    return {
        "pct_computer": task_features_df["has_computer"].mean(),
        "pct_physical": task_features_df["is_physical"].mean(),
        "pct_communication": task_features_df["must_communicate"].mean(),
        "pct_analyze": task_features_df["must_analyze"].mean(),
        "pct_manage": task_features_df["must_manage"].mean(),
        "pct_creative": task_features_df["is_creative"].mean(),
        "pct_textnative": task_features_df["is_text_native"].mean(),
    }


# ======================================================================
# Fallback imputation by major group
# ======================================================================

def compute_fallback_stats(dataset):
    """
    precompute major group medians/modes for imputing missing values.
    mirrors data-transform notebook: numeric -> median, binary -> mode.
    call once at app startup.
    """
    dataset = dataset.copy()
    dataset["major_group"] = dataset["occ_code"].str[:2]

    stats = {}
    for mg in dataset["major_group"].unique():
        group = dataset[dataset["major_group"] == mg]
        stats[mg] = {
            "MedianSalary": group["MedianSalary"].median(),
            "JobZone": round(group["JobZone"].median()),
            "isBright": int(group["isBright"].mode().iloc[0]) if not group["isBright"].mode().empty else 0,
            "isGreen": int(group["isGreen"].mode().iloc[0]) if not group["isGreen"].mode().empty else 0,
        }

    stats["global"] = {
        "MedianSalary": dataset["MedianSalary"].median(),
        "JobZone": round(dataset["JobZone"].median()),
        "isBright": 0,
        "isGreen": 0,
    }

    return stats


def _get_fallback(occ_code, fallback_stats):
    """get fallback values for the occ_code's major group"""
    major_group = occ_code[:2]
    return fallback_stats.get(major_group, fallback_stats["global"])


# ======================================================================
# Feature vector assembly
# ======================================================================

def _build_feature_vector(is_bright, is_green, job_zone, median_salary, task_pcts):
    """assemble the 11-feature row matching model's training column order"""
    features = {
        "isBright": int(is_bright),
        "isGreen": int(is_green),
        "JobZone": job_zone,
        "MedianSalary": median_salary,
        **task_pcts
    }
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


# ======================================================================
# SOC code resolution
# ======================================================================

def _resolve_soc_code(job_title, job_description):
    """claude generates keywords -> onet search -> claude picks best match"""
    keywords = _generate_keywords(job_title, job_description)
    if keywords is None:
        return "INVALID_JOB_TITLE", None, None
    candidates = _search_onet(keywords)

    if not candidates:
        return None, None, None

    soc_code = _select_best_occupation(job_title, job_description, candidates)
    title = candidates.get(soc_code, {}).get("title", job_title)

    return soc_code, title, candidates


# ======================================================================
# Main prediction functions (called by the app)
# ======================================================================

def predict_ai_job_exposure(job_title, model, scaler, dataset, fallback_stats, job_description=None):
    """
    full pipeline:
    1. resolve job title -> soc_code via claude + onet
    2. check if occ_code exists in dataset -> return stored prediction
    3. if not -> fetch from onet (tasks, job zone, flags)
              -> fetch salary from bls (fall back to major group median)
              -> engineer features -> predict
    """

    if not job_title or not isinstance(job_title, str) or not job_title.strip():
        return {"error": "Job title is required and must be a non-empty string"}

    print(f"\n{'='*60}")
    print(f"[pipeline] predict_ai_job_exposure ← {repr(job_title)}")
    if job_description:
        print(
            f"[pipeline] description: {repr(job_description[:100])}{'...' if len(job_description) > 100 else ''}")

    # step 1: resolve to soc code
    soc_code, onet_title, candidates = _resolve_soc_code(
        job_title, job_description)

    if soc_code == "INVALID_JOB_TITLE":
        return {"error": "INVALID_JOB_TITLE"}

    if soc_code is None:
        return {"error": f"No O*NET occupations found for '{job_title}'"}

    # step 2: check dataset ("15-1252.00" -> "15-1252")
    occ_code_short = soc_code.split(".")[0]
    match = dataset[dataset["occ_code"] == occ_code_short]

    if not match.empty:
        row = match.iloc[0]
        X = pd.DataFrame([row[FEATURE_COLUMNS].values],
                         columns=FEATURE_COLUMNS)
        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0])
        print(
            f"[pipeline] source=dataset  occ={row['occ_code']}  title={repr(row['title'])}")
        print(
            f"[pipeline] prediction={prediction}  probability={probability:.3f}")
        print(f"{'='*60}\n")

        return {
            "source": "dataset",
            "occ_code": row["occ_code"],
            "title": row["title"],
            "prediction": prediction,
            "probability": probability,
            "features": X.iloc[0].to_dict()
        }

    # step 3: not in dataset — fetch from apis
    tasks = _fetch_tasks(soc_code)
    job_zone = _fetch_job_zone(soc_code)
    flags = _fetch_occupation_flags(soc_code)

    # task features
    if tasks:
        task_df = _keyword_features(tasks)
        task_pcts = _aggregate_task_features(task_df)
    else:
        task_pcts = {
            col: 0.0 for col in FEATURE_COLUMNS if col.startswith("pct_")}

    # fallback values for missing fields
    fallback = _get_fallback(occ_code_short, fallback_stats)

    is_bright = flags["isBright"] if flags["isBright"] is not None else fallback["isBright"]
    is_green = flags["isGreen"] if flags["isGreen"] is not None else fallback["isGreen"]
    job_zone = job_zone if job_zone is not None else fallback["JobZone"]

    # salary: bls -> major group median
    median_salary = _fetch_bls_median_salary(soc_code)
    salary_source = "bls"
    if median_salary is None:
        median_salary = fallback["MedianSalary"]
        salary_source = "fallback"

    # build, scale, predict
    X = _build_feature_vector(
        is_bright, is_green, job_zone, median_salary, task_pcts)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0])
    print(
        f"[pipeline] source=onet_api  occ={occ_code_short}  title={repr(onet_title)}")
    print(f"[pipeline] salary_source={salary_source}  tasks={len(tasks)}")
    print(f"[pipeline] prediction={prediction}  probability={probability:.3f}")
    print(f"{'='*60}\n")

    return {
        "source": "onet_api",
        "occ_code": occ_code_short,
        "title": onet_title,
        "prediction": prediction,
        "probability": probability,
        "features": X.iloc[0].to_dict(),
        "tasks_retrieved": len(tasks),
        "salary_source": salary_source
    }


def predict_manual(features_dict, model, scaler):
    missing = [col for col in FEATURE_COLUMNS if col not in features_dict]

    if missing:
        return {"error": f"Missing features: {missing}"}

    for col in FEATURE_COLUMNS:
        val = features_dict[col]
        if not isinstance(val, (int, float)):
            return {"error": f"Feature '{col}' must be numeric, got {type(val).__name__}"}

    # range checks
    if features_dict["isBright"] not in (0, 1):
        return {"error": "isBright must be 0 or 1"}
    if features_dict["isGreen"] not in (0, 1):
        return {"error": "isGreen must be 0 or 1"}
    if not (1 <= features_dict["JobZone"] <= 5):
        return {"error": "JobZone must be between 1 and 5"}
    if features_dict["MedianSalary"] < 0:
        return {"error": "MedianSalary cannot be negative"}
    for col in FEATURE_COLUMNS:
        if col.startswith("pct_") and not (0 <= features_dict[col] <= 1):
            return {"error": f"{col} must be between 0 and 1"}

    """scenario 3: predict from manually adjusted feature values (sliders/toggles)"""
    X = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)

    return {
        "prediction": int(model.predict(X_scaled)[0]),
        "probability": float(model.predict_proba(X_scaled)[0]),
        "features": features_dict
    }
