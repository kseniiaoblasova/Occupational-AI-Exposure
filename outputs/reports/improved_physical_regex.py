"""
Improved is_physical regex — drop-in replacement for pipeline.py and data-transform notebook.

TRADEOFF SUMMARY (tested against 18,428 unique O*NET task statements):
                                            OLD         NEW
Overall match rate                         45.4%       23.5%
Office/IT/Legal/Education false positives  25.7%        5.5%
Construction/Maint/Prod/Transport TP       71.6%       50.5%
Healthcare (mixed physical)                43.4%       13.4%

The old regex used bare substring matches (e.g. 'car' matching 'career',
'open' matching 'open communication', 'run' matching 'prune', 'add' matching
'ladder'). The new regex uses word boundaries (\\b) on standalone physical
verbs that are unambiguous, and compound patterns (verb + physical context
object) for verbs that are ambiguous on their own (e.g. 'cut' only matches
when followed by a physical material like 'metal', 'wood', 'pipe').

The true positive rate drop (71.6% -> 50.5%) is a deliberate precision-recall
tradeoff. The missed physical tasks use generic verbs like 'monitor', 'mark',
'string', 'regulate', 'activate' that also appear heavily in office/analytical
contexts. Flagging these would bring false positives back above 15%.

HOW TO USE: Replace the is_physical pattern in both:
  - notebooks/data-transform.ipynb (cell 19)
  - src/pipeline.py (_keyword_features function, line ~239)
Then retrain the model, since the pct_physical feature values will change.
"""

# Drop-in replacement — paste this block where the old is_physical assignment is

task_features_by_occ['is_physical'] = task_features_by_occ['task_name'].str.contains(
    # --- standalone physical verbs (safe with word boundaries) ---
    r'\brepair(?:ing|s|ed)?\b|\bassembl(?:e|ing|y|ies|ed)\b|\binstall(?:ing|ation|s|ed)?\b'
    r'|\bload(?:ing|s|ed)?\b|\bunload(?:ing|s|ed)?\b|\blift(?:ing|s|ed)?\b'
    r'|\bweld(?:ing|s|ed)?\b|\bdrill(?:ing|s|ed)?\b|\bgrind(?:ing|s)?\b'
    r'|\bshovel(?:ing|s)?\b|\bexcavat(?:e|ing|ion)\b'
    r'|\bdemolish(?:ing|es|ed)?\b|\bconstruct(?:ing|ion|s|ed)?\b'
    r'|\bfasten(?:ing|s|ed|er)?\b|\bbolt(?:ing|s|ed)?\b|\bsolder(?:ing|s|ed)?\b'
    r'|\bhoist(?:ing|s|ed)?\b|\bhaul(?:ing|s|ed)?\b'
    r'|\bpav(?:e|ing|ed)\b|\bplumb(?:ing)?\b|\bdrywall(?:ing)?\b'
    r'|\bcement(?:ing)?\b|\bbrick(?:lay|laying|s)?\b'
    r'|\bscaffold(?:ing)?\b|\broof(?:ing)?\b|\binsulat(?:e|ing|ion)\b'
    r'|\bvacuum(?:ing|s)?\b|\bsweep(?:ing|s)?\b|\bmop(?:ping|s)?\b|\bscrub(?:bing|s)?\b'
    r'|\bwash(?:ing|es|ed)?\b|\bwipe(?:s|d)?\b|\bpolish(?:ing|es|ed)?\b'
    r'|\bsaniti(?:ze|zing|zed)\b|\bsteriliz(?:e|ing|ed)\b'
    r'|\bclean(?:ing|s|ed)?\b|\bmow(?:ing|s|ed)?\b|\bprun(?:e|ing|ed)\b'
    r'|\blandscap(?:e|ing|ed)\b|\bfertiliz(?:e|ing|ed)\b|\bfumigat(?:e|ing|ed)\b|\birrigat(?:e|ing|ed)\b'
    r'|\bharvest(?:ing|s|ed)?\b|\bslaughter(?:ing|s|ed)?\b|\bbutcher(?:ing|s|ed)?\b'
    r'|\bcook(?:ing|s|ed)?\b|\bbak(?:e|ing|es|ed)\b|\bgrill(?:ing|s|ed)?\b|\bfry(?:ing|ies|ied)?\b'
    r'|\bcarry(?:ing|ies)?\b|\bstack(?:ing|s|ed)?\b'
    r'|\bpatrol(?:ling|s|led)?\b|\brescue\b|\bextinguish(?:ing|ed)?\b|\bfirefight'
    r'|\bsew(?:ing|s|n|ed)?\b|\bstitch(?:ing|es|ed)?\b|\bknit(?:ting|s|ted)?\b|\bweav(?:e|ing|es|ed)\b'
    r'|\bglue\b|\bgluing\b|\bwax(?:ing)?\b'
    r'|\bmassag(?:e|ing)\b|\btow(?:ing|s|ed)?\b'
    r'|\b(?:arrest|apprehend|detain|handcuff|frisk)\b'
    r'|\bsplint(?:ing|s|ed)?\b|\bbandage\b|\bimmobili\w+'
    r'|\b(?:bathe|groom|toileting)\b'
    r'|\bhand\s+tool|\bpower\s+tool'
    r'|\bdisassembl(?:e|ing|y|ed)\b'
    r'|\blubricat(?:e|ing|ed|ion)\b'
    # --- operate/tend + physical equipment ---
    r'|\b(?:operat|tend)\w*\b.*\b(?:machine|equipment|vehicle|forklift|crane|saw|lathe|'
    r'press|pump|valve|boiler|furnace|tractor|excavator|loader|bulldozer|conveyor|'
    r'compressor|generator|mixer|welder|grinder|cutter|sewing|shear|router|sander|'
    r'milling|stamping|printing|kiln|oven|dredge|winch)\b'
    # --- machine + physical verb context (bidirectional) ---
    r'|\bmachine(?:ry|s)?\b.*\b(?:operat|tend|set\s*up|adjust|calibrat|maintain|feed|'
    r'thread|align|repair|clean|lubricat)\b'
    r'|\b(?:operat|tend|set\s*up|adjust|calibrat|maintain|feed|thread|align|repair|'
    r'clean|lubricat)\w*\b.*\bmachine(?:ry|s)?\b'
    # --- drive + vehicle ---
    r'|\bdriv(?:e|ing|es)\b.*\b(?:vehicle|truck|bus|car|forklift|tractor|ambulance|'
    r'route|taxi|van|locomotive|boat|ship)\b'
    # --- transport + physical objects/people ---
    r'|\btransport(?:ing|s|ed)?\b.*\b(?:patient|material|equipment|goods|cargo|'
    r'passenger|product|supplie|item|specimen|sample|prisoner|detainee)\b'
    # --- deliver + physical items ---
    r'|\bdeliver(?:ing|s|ed|y|ies)?\b.*\b(?:mail|package|goods|product|material|'
    r'supplie|food|cargo|item|order|part|newspaper|furniture)\b'
    # --- serve + food context ---
    r'|\bserv(?:e|ing|es|ed)\b.*\b(?:food|meal|drink|beverage|dish|plate|snack)\b'
    # --- cut + physical materials ---
    r'|\bcut(?:ting|s)?\b.*\b(?:material|metal|wood|lumber|pipe|wire|glass|fabric|'
    r'cloth|tile|stone|meat|hair|hedge|grass|paper|foam|leather|plastic|tubing|sheet|'
    r'board|carpet|drywall|bolt|thread|cable|rope|strap|chain|hose|bar)\b'
    # --- pour + physical substances ---
    r'|\bpour(?:ing|s|ed)?\b.*\b(?:concrete|metal|liquid|molten|mixture|resin|mold|'
    r'foundation|water|chemical|plaster|asphalt|cement|slurry|batter|wax)\b'
    # --- mount + physical objects ---
    r'|\bmount(?:ing|s|ed)?\b.*\b(?:equipment|component|part|device|wheel|tire|display|'
    r'panel|bracket|antenna|fixture|motor|engine|lens|specimen|plate|gun|blade|negative|'
    r'mold|sign|frame|bearing)\b'
    # --- spray + substances ---
    r'|\bspray(?:ing|s|ed)?\b.*\b(?:paint|pesticide|herbicide|insecticide|chemical|'
    r'coating|foam|water|finish|adhesive|material|concrete|sealant|lacquer|enamel|'
    r'primer|stain)\b'
    # --- mix + physical substances ---
    r'|\bmix(?:ing|es|ed)?\b.*\b(?:concrete|mortar|plaster|paint|chemical|ingredient|'
    r'compound|batch|solution|formula|resin|adhesive|dye|cement|grout|epoxy|batter|'
    r'dough|feed|fertiliz|powder|sand|clay|gravel)\b'
    # --- measure + physical dimensions/objects ---
    r'|\bmeasur(?:e|ing|es|ed)\b.*\b(?:dimension|length|width|height|depth|distance|'
    r'angle|clearance|tolerance|gap|fit|material|pipe|wire|board|lumber|fabric|glass|'
    r'tile|part|customer|body|patient|room|area|opening|specimen|sample)\b'
    # --- remove + physical objects ---
    r'|\bremov(?:e|ing|es|ed)\b.*\b(?:part|component|debris|waste|stain|mold|asbestos|'
    r'snow|ice|rust|old|damaged|worn|defective|tire|wheel|door|panel|fixture|nail|screw|'
    r'bolt|pin|bearing|gasket|seal|filter|hose|belt|insulation|flooring|shingle|tile|'
    r'bark|limb|stump|organ|tissue|foreign|splint|cast|staple|suture|drum|spindle|'
    r'bobbin|excess|burr|flash|slag)\b'
    # --- adjust + physical equipment ---
    r'|\badjust(?:ing|s|ed)?\b.*\b(?:machine|equipment|tool|valve|belt|chain|brake|'
    r'clutch|throttle|dial|setting|tension|pressure|alignment|calibrat|speed|temperature|'
    r'flow|height|blade|guide|gauge|control|clamp|fixture|jig|mold|die|plate|roller|'
    r'wheel|mirror|antenna|seat)\b'
    # --- open/close + physical objects ---
    r'|\b(?:open|close)\b.*\b(?:valve|hatch|lid|door|gate|window|drawer|cabinet|mold|'
    r'container|tank|switch|circuit|breaker|cock|nozzle|damper|shutter|vent|plug|cap|clamp)\b'
    # --- attach + physical objects ---
    r'|\battach(?:ing|es|ed)?\b.*\b(?:part|component|wire|cable|pipe|hose|fitting|bracket|'
    r'clamp|strap|label|spring|fixture|device|hardware|panel|bolt|screw|rivet|clip|hook|'
    r'chain|rope|handle|knob|antenna|electrode|lead|terminal|tube|nozzle|coupling|flange|'
    r'plate|frame)\b'
    # --- wire/wiring + electrical context ---
    r'|\bwir(?:e|ing|ed)\b.*\b(?:circuit|electrical|connect|panel|outlet|switch|motor|'
    r'system|component|control|terminal|harness|junction|transformer|relay|sensor)\b'
    # --- seal + physical context ---
    r'|\bseal(?:ing|ant|s|ed)?\b.*\b(?:joint|pipe|surface|container|crack|gap|seam|duct|'
    r'window|door|roof|tank|envelope|edge|connection|flange|valve|opening|cylinder|leak)\b'
    # --- store + physical items ---
    r'|\bstore\b.*\b(?:material|equipment|tool|supplie|product|item|chemical|hazardous|'
    r'food|grain|hay|weapon|ammunition|inventory|part|goods|fuel|sample|specimen|explosive)\b'
    # --- sort + physical items ---
    r'|\bsort(?:ing|s|ed)?\b.*\b(?:mail|package|item|material|product|part|laundry|'
    r'recycl|waste|lumber|seed|fruit|vegetable|fish|egg|stone|ore|scrap|garment|fabric)\b'
    # --- plant + growing context ---
    r'|\bplant(?:ing|s|ed)?\b.*\b(?:seed|crop|tree|bulb|flower|shrub|vegetable|grass|'
    r'vine|herb|seedling)\b'
    # --- dig + physical context ---
    r'|\bdig(?:ging|s)?\b.*\b(?:hole|trench|ditch|foundation|grave|post|well|channel|'
    r'pit|tunnel|earth|soil|ground)\b'
    # --- guard + physical location ---
    r'|\bguard(?:ing|s|ed)?\b.*\b(?:area|building|facility|premise|entrance|gate|'
    r'perimeter|prisoner|property|site|door|access|border|inmate)\b'
    # --- push/pull + physical objects ---
    r'|\bpush(?:ing|es|ed)?\b.*\b(?:cart|wheelchair|gurney|stretcher|dolly|mower|plow|'
    r'broom|material|object|vehicle|equipment)\b'
    r'|\bpull(?:ing|s|ed)?\b.*\b(?:cable|wire|rope|hose|pipe|lever|cart|chain|strap|'
    r'cord|thread|string|nail|tooth|weed)\b'
    # --- physical exams / medical procedures ---
    r'|\bphysical\s+exam|\bexamine\b.*\b(?:patient|client)\b'
    r'|\bmanipulat(?:e|ing)\b.*\b(?:joint|spine|tissue|muscle|body)\b'
    r'|\bperform\b.*\bsurger|\bsurgical\s+procedure'
    r'|\bvital\s+sign|\bblood\s+pressure'
    r'|\b(?:administer|inject|infuse|insert|suture|incision|intubat)\w*\b.*\b(?:patient|'
    r'medic|drug|treatment|anesthes|blood|fluid|needle|catheter|vaccine|shot|dose|iv\b)\b'
    r'|\bdress(?:ing)?\b.*\bwound'
    # --- escort ---
    r'|\bescort(?:ing)?\b.*\b(?:patient|client|resident|prisoner|detainee|visitor)\b'
    # --- fitting physical devices ---
    r'|\b(?:fit|fitting)\b.*\b(?:prosthe|orthotic|hearing\s+aid|eyeglass|lens|brace|'
    r'denture|crown|implant)\b'
    # --- cover + physical surfaces ---
    r'|\bcover(?:ing)?\b.*\b(?:surface|window|floor|wall|roof|pipe|hole|wound|opening|'
    r'furniture|vehicle)\b'
    # --- signal workers ---
    r'|\bsignal\b.*\b(?:worker|operator|driver|pilot|crane|equipment)\b'
    # --- connect/disconnect physical items ---
    r'|\b(?:connect|disconnect)(?:ing|s|ed)?\b.*\b(?:pipe|hose|wire|cable|tube|coupling|'
    r'fitting|electrode|terminal|line|cord|duct|valve|meter|gauge|sensor|device)\b'
    # --- position physical objects ---
    r'|\bposition(?:ing|s|ed)?\b.*\b(?:part|component|material|workpiece|mold|die|plate|'
    r'fixture|clamp|patient|equipment|antenna|beam|column|pipe|panel|frame|block|stone|brick)\b'
    # --- feed machines/animals ---
    r'|\bfeed(?:ing|s)?\b.*\b(?:machine|press|hopper|conveyor|roller|mill|grinder|cutter|'
    r'shredder|animal|livestock|patient|infant)\b'
    # --- thread physical objects ---
    r'|\bthread(?:ing|s|ed)?\b.*\b(?:needle|machine|bolt|pipe|wire|reel|bobbin|spool|'
    r'film|tape)\b'
    # --- set up physical equipment ---
    r'|\bset\s+up\b.*\b(?:machine|equipment|scaffold|stage|display|booth|tent|table|'
    r'barrier|cone|sign|ladder|tool|jig|fixture|mold|die|press|lathe|drill|saw|grinder)\b'
    # --- wrap physical items ---
    r'|\bwrap(?:ping|s|ped)?\b.*\b(?:package|gift|product|item|material|bandage|wire|'
    r'cable|pipe|insulation|food|meat|pallet|box|load|bundle)\b',
    case=False
).astype(int)
