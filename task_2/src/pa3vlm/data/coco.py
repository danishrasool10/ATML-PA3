"""COCO 2014 helpers: category wording table, annotation indexing, caption mention check."""
from __future__ import annotations

import re
from collections import defaultdict

# COCO name -> (presence phrase incl. article, counting plural, extra caption synonyms)
# Display names avoid ambiguity ("computer mouse", "remote control"); synonyms are used only to reject
# presence negatives whose captions mention the object (COCO annotation omissions).
CATEGORIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "person": ("a person", "people", ("person", "people", "persons", "man", "men", "woman", "women", "boy", "boys",
               "girl", "girls", "child", "children", "kid", "kids", "player", "players", "guy", "guys", "lady",
               "ladies", "baby", "babies", "skier", "skiers", "surfer", "surfers", "skateboarder", "snowboarder",
               "rider", "crowd", "someone", "toddler", "adult", "adults", "pitcher", "batter", "catcher", "umpire")),
    "bicycle": ("a bicycle", "bicycles", ("bicycle", "bicycles", "bike", "bikes", "cyclist")),
    "car": ("a car", "cars", ("car", "cars", "taxi", "suv", "sedan")),
    "motorcycle": ("a motorcycle", "motorcycles", ("motorcycle", "motorcycles", "motorbike", "motor bike", "scooter",
                   "moped", "dirt bike")),
    "airplane": ("an airplane", "airplanes", ("airplane", "airplanes", "plane", "planes", "jet", "jets", "aircraft",
                 "airliner")),
    "bus": ("a bus", "buses", ("bus", "buses", "busses")),
    "train": ("a train", "trains", ("train", "trains", "locomotive", "tram", "trolley", "subway")),
    "truck": ("a truck", "trucks", ("truck", "trucks", "pickup", "lorry")),
    "boat": ("a boat", "boats", ("boat", "boats", "ship", "ships", "canoe", "kayak", "sailboat", "yacht", "ferry")),
    "traffic light": ("a traffic light", "traffic lights", ("traffic light", "traffic lights", "stoplight",
                      "stop light", "traffic signal")),
    "fire hydrant": ("a fire hydrant", "fire hydrants", ("hydrant", "hydrants")),
    "stop sign": ("a stop sign", "stop signs", ("stop sign", "stop signs")),
    "parking meter": ("a parking meter", "parking meters", ("parking meter", "parking meters", "meter", "meters")),
    "bench": ("a bench", "benches", ("bench", "benches")),
    "bird": ("a bird", "birds", ("bird", "birds", "duck", "ducks", "pigeon", "pigeons", "seagull", "gull", "goose",
             "geese", "parrot", "swan", "owl", "eagle")),
    "cat": ("a cat", "cats", ("cat", "cats", "kitten", "kitty", "kittens")),
    "dog": ("a dog", "dogs", ("dog", "dogs", "puppy", "puppies", "pup")),
    "horse": ("a horse", "horses", ("horse", "horses", "pony", "foal")),
    "sheep": ("a sheep", "sheep", ("sheep", "lamb", "lambs", "ram")),
    "cow": ("a cow", "cows", ("cow", "cows", "cattle", "bull", "bulls", "calf", "ox", "oxen")),
    "elephant": ("an elephant", "elephants", ("elephant", "elephants")),
    "bear": ("a bear", "bears", ("bear", "bears")),
    "zebra": ("a zebra", "zebras", ("zebra", "zebras")),
    "giraffe": ("a giraffe", "giraffes", ("giraffe", "giraffes")),
    "backpack": ("a backpack", "backpacks", ("backpack", "backpacks", "back pack", "rucksack")),
    "umbrella": ("an umbrella", "umbrellas", ("umbrella", "umbrellas", "parasol")),
    "handbag": ("a handbag", "handbags", ("handbag", "handbags", "purse", "purses", "bag", "bags")),
    "tie": ("a tie", "ties", ("tie", "ties", "necktie", "bow tie")),
    "suitcase": ("a suitcase", "suitcases", ("suitcase", "suitcases", "luggage", "baggage")),
    "frisbee": ("a frisbee", "frisbees", ("frisbee", "frisbees", "disc")),
    "skis": ("skis", "pairs of skis", ("ski", "skis", "skiing", "skier", "skiers")),
    "snowboard": ("a snowboard", "snowboards", ("snowboard", "snowboards", "snowboarding", "snowboarder")),
    "sports ball": ("a ball", "balls", ("ball", "balls", "soccer ball", "football", "baseball", "tennis ball",
                    "basketball", "volleyball")),
    "kite": ("a kite", "kites", ("kite", "kites")),
    "baseball bat": ("a baseball bat", "baseball bats", ("bat", "bats")),
    "baseball glove": ("a baseball glove", "baseball gloves", ("glove", "gloves", "mitt", "mitts")),
    "skateboard": ("a skateboard", "skateboards", ("skateboard", "skateboards", "skateboarding", "skateboarder")),
    "surfboard": ("a surfboard", "surfboards", ("surfboard", "surfboards", "surf board", "surfing", "surfer",
                  "board")),
    "tennis racket": ("a tennis racket", "tennis rackets", ("racket", "rackets", "racquet", "racquets")),
    "bottle": ("a bottle", "bottles", ("bottle", "bottles")),
    "wine glass": ("a wine glass", "wine glasses", ("wine glass", "wine glasses", "glass of wine", "wineglass")),
    "cup": ("a cup", "cups", ("cup", "cups", "mug", "mugs", "coffee")),
    "fork": ("a fork", "forks", ("fork", "forks")),
    "knife": ("a knife", "knives", ("knife", "knives")),
    "spoon": ("a spoon", "spoons", ("spoon", "spoons")),
    "bowl": ("a bowl", "bowls", ("bowl", "bowls")),
    "banana": ("a banana", "bananas", ("banana", "bananas")),
    "apple": ("an apple", "apples", ("apple", "apples")),
    "sandwich": ("a sandwich", "sandwiches", ("sandwich", "sandwiches", "burger", "hamburger", "sub")),
    "orange": ("an orange", "oranges", ("orange", "oranges", "citrus", "tangerine")),
    "broccoli": ("broccoli", "pieces of broccoli", ("broccoli",)),
    "carrot": ("a carrot", "carrots", ("carrot", "carrots")),
    "hot dog": ("a hot dog", "hot dogs", ("hot dog", "hot dogs", "hotdog", "hotdogs")),
    "pizza": ("a pizza", "pizzas", ("pizza", "pizzas")),
    "donut": ("a donut", "donuts", ("donut", "donuts", "doughnut", "doughnuts")),
    "cake": ("a cake", "cakes", ("cake", "cakes", "cupcake", "cupcakes")),
    "chair": ("a chair", "chairs", ("chair", "chairs", "seat", "seats", "stool")),
    "couch": ("a couch", "couches", ("couch", "couches", "sofa", "sofas", "loveseat")),
    "potted plant": ("a potted plant", "potted plants", ("plant", "plants", "flower", "flowers", "pot", "potted")),
    "bed": ("a bed", "beds", ("bed", "beds", "bedroom")),
    "dining table": ("a dining table", "dining tables", ("table", "tables", "desk")),
    "toilet": ("a toilet", "toilets", ("toilet", "toilets", "urinal")),
    "tv": ("a TV", "TVs", ("tv", "tvs", "television", "televisions", "monitor", "screen")),
    "laptop": ("a laptop", "laptops", ("laptop", "laptops", "computer", "computers", "notebook")),
    "mouse": ("a computer mouse", "computer mice", ("mouse", "mice")),
    "remote": ("a remote control", "remote controls", ("remote", "remotes", "controller", "controllers")),
    "keyboard": ("a keyboard", "keyboards", ("keyboard", "keyboards")),
    "cell phone": ("a cell phone", "cell phones", ("phone", "phones", "cellphone", "cellphones", "smartphone",
                   "mobile")),
    "microwave": ("a microwave", "microwaves", ("microwave", "microwaves")),
    "oven": ("an oven", "ovens", ("oven", "ovens", "stove", "stoves")),
    "toaster": ("a toaster", "toasters", ("toaster", "toasters")),
    "sink": ("a sink", "sinks", ("sink", "sinks")),
    "refrigerator": ("a refrigerator", "refrigerators", ("refrigerator", "refrigerators", "fridge", "fridges")),
    "book": ("a book", "books", ("book", "books", "bookshelf", "novel")),
    "clock": ("a clock", "clocks", ("clock", "clocks")),
    "vase": ("a vase", "vases", ("vase", "vases")),
    "scissors": ("scissors", "pairs of scissors", ("scissor", "scissors")),
    "teddy bear": ("a teddy bear", "teddy bears", ("teddy", "teddies", "stuffed animal", "stuffed bear",
                   "stuffed animals")),
    "hair drier": ("a hair dryer", "hair dryers", ("hair dryer", "hair drier", "blow dryer", "hairdryer")),
    "toothbrush": ("a toothbrush", "toothbrushes", ("toothbrush", "toothbrushes", "tooth brush")),
}
# Categories whose instance counts are ambiguous (pairs, bunches, uncountable, stacks): no counting questions.
NO_COUNTING = {"skis", "banana", "broccoli", "carrot", "book", "scissors"}


def presence_question(cat: str) -> str:
    return f"Does this image contain {CATEGORIES[cat][0]}?"


def counting_question(cat: str) -> str:
    return f"How many {CATEGORIES[cat][1]} can you see?"


def _patterns(cat: str) -> re.Pattern:
    words = set(CATEGORIES[cat][2]) | {cat}
    alt = "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)


_PAT_CACHE: dict[str, re.Pattern] = {}


def captions_mention(cat: str, captions: list[str]) -> bool:
    if cat not in _PAT_CACHE:
        _PAT_CACHE[cat] = _patterns(cat)
    pat = _PAT_CACHE[cat]
    return any(pat.search(c) for c in captions)


def index_captions(caps_json: dict) -> dict[int, list[dict]]:
    """image_id -> caption annotations sorted by annotation id."""
    out: dict[int, list[dict]] = defaultdict(list)
    for a in caps_json["annotations"]:
        out[a["image_id"]].append({"id": a["id"], "caption": a["caption"]})
    for v in out.values():
        v.sort(key=lambda r: r["id"])
    return dict(out)


def index_instances(inst_json: dict) -> tuple[dict[int, str], dict[int, dict[str, dict]]]:
    """Returns (category_id -> name, image_id -> {name: {n, n_crowd, min_area, max_area}}).

    Images without any instance annotation map to {} (all COCO images are listed)."""
    cats = {c["id"]: c["name"] for c in inst_json["categories"]}
    assert set(cats.values()) == set(CATEGORIES), "unexpected COCO category set"
    stats: dict[int, dict[str, dict]] = {im["id"]: {} for im in inst_json["images"]}
    for a in inst_json["annotations"]:
        s = stats.setdefault(a["image_id"], {}).setdefault(
            cats[a["category_id"]], {"n": 0, "n_crowd": 0, "min_area": float("inf"), "max_area": 0.0})
        if a.get("iscrowd", 0):
            s["n_crowd"] += 1
        else:
            s["n"] += 1
            s["min_area"] = min(s["min_area"], float(a["area"]))
            s["max_area"] = max(s["max_area"], float(a["area"]))
    return cats, stats


def presence_positive_ok(st: dict[str, dict], cat: str, min_area: float) -> bool:
    s = st.get(cat)
    return bool(s) and s["n"] >= 1 and s["max_area"] >= min_area


def presence_negative_ok(st: dict[str, dict], cat: str, captions: list[str]) -> bool:
    return cat not in st and len(st) > 0 and not captions_mention(cat, captions)


def count_ok(st: dict[str, dict], cat: str, min_area: float, max_count: int) -> int | None:
    """Unambiguous count in 1..max_count, else None."""
    if cat in NO_COUNTING:
        return None
    s = st.get(cat)
    if not s or s["n_crowd"] > 0 or not (1 <= s["n"] <= max_count) or s["min_area"] < min_area:
        return None
    return s["n"]
