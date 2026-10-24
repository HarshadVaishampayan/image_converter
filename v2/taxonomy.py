"""Tag vocabulary for zero-shot image tagging.

Each label maps to (parents, aliases). The image embedding is scored against
every label; a confident specific label (e.g. "cherry blossom") also implies
its parents ("flower", "tree", "plant"), so broad searches like "flowers" find
images whose embeddings only strongly match a specific kind of flower.

Labels are matched as written in prompts ("a photo of {label}"), so keep them
in natural singular form. Parents must themselves be labels.
"""

TAXONOMY = {
    # ---- plants & flowers ----
    "plant": ((), ["greenery", "foliage"]),
    "flower": (("plant",), ["bloom", "blossom", "floral"]),
    "rose": (("flower",), []),
    "cherry blossom": (("flower", "tree"), ["sakura", "cherry tree"]),
    "tulip": (("flower",), []),
    "sunflower": (("flower",), []),
    "daisy": (("flower",), []),
    "orchid": (("flower",), []),
    "lily": (("flower",), []),
    "lotus": (("flower",), ["water lily"]),
    "lavender": (("flower",), []),
    "hibiscus": (("flower",), []),
    "dandelion": (("flower",), []),
    "poppy": (("flower",), []),
    "magnolia": (("flower", "tree"), []),
    "hydrangea": (("flower",), []),
    "peony": (("flower",), []),
    "marigold": (("flower",), []),
    "chrysanthemum": (("flower",), ["mum"]),
    "bouquet": (("flower",), ["flower arrangement"]),
    "flower field": (("flower", "field"), ["flower bed", "garden bed"]),
    "tree": (("plant",), []),
    "palm tree": (("tree",), ["palm"]),
    "pine tree": (("tree",), ["conifer", "evergreen"]),
    "autumn leaves": (("tree", "autumn"), ["fall foliage", "fall colors"]),
    "forest": (("tree", "nature"), ["woods", "woodland"]),
    "cactus": (("plant",), ["succulent"]),
    "grass": (("plant",), ["lawn"]),
    "leaf": (("plant",), []),
    "mushroom": (("nature",), ["fungus"]),
    "garden": (("plant",), []),

    # ---- animals ----
    "animal": ((), ["creature", "wildlife", "pet"]),
    "dog": (("animal",), ["puppy", "canine"]),
    "cat": (("animal",), ["kitten", "feline"]),
    "bird": (("animal",), []),
    "parrot": (("bird",), []),
    "owl": (("bird",), []),
    "duck": (("bird",), []),
    "eagle": (("bird",), []),
    "penguin": (("bird",), []),
    "chicken": (("bird",), ["hen", "rooster"]),
    "horse": (("animal",), ["pony"]),
    "cow": (("animal",), ["cattle"]),
    "sheep": (("animal",), ["lamb"]),
    "pig": (("animal",), []),
    "goat": (("animal",), []),
    "rabbit": (("animal",), ["bunny"]),
    "squirrel": (("animal",), []),
    "deer": (("animal",), []),
    "bear": (("animal",), []),
    "elephant": (("animal",), []),
    "lion": (("animal",), []),
    "tiger": (("animal",), []),
    "monkey": (("animal",), ["ape"]),
    "giraffe": (("animal",), []),
    "zebra": (("animal",), []),
    "fish": (("animal",), []),
    "butterfly": (("insect",), []),
    "bee": (("insect",), []),
    "insect": (("animal",), ["bug"]),
    "spider": (("animal",), []),
    "snake": (("animal",), ["reptile"]),
    "turtle": (("animal",), ["tortoise"]),
    "frog": (("animal",), []),

    # ---- people ----
    "person": ((), ["people", "human"]),
    "child": (("person",), ["kid", "girl", "boy"]),
    "baby": (("person",), ["infant", "toddler"]),
    "portrait": (("person",), ["face", "headshot"]),
    "selfie": (("person", "portrait"), []),
    "group of people": (("person",), ["crowd", "family", "friends"]),
    "wedding": (("person", "event"), ["bride", "groom"]),

    # ---- food & drink ----
    "food": ((), ["meal", "dish"]),
    "pizza": (("food",), []),
    "burger": (("food",), ["hamburger"]),
    "sandwich": (("food",), []),
    "pasta": (("food",), ["spaghetti", "noodles"]),
    "sushi": (("food",), []),
    "salad": (("food",), []),
    "soup": (("food",), []),
    "cake": (("food", "dessert"), []),
    "dessert": (("food",), ["sweets"]),
    "ice cream": (("food", "dessert"), []),
    "bread": (("food",), []),
    "fruit": (("food",), []),
    "apple": (("fruit",), []),
    "banana": (("fruit",), []),
    "strawberry": (("fruit",), []),
    "orange": (("fruit",), []),
    "vegetable": (("food",), ["vegetables", "veggies"]),
    "breakfast": (("food",), []),
    "coffee": (("drink",), ["latte", "espresso"]),
    "tea": (("drink",), []),
    "wine": (("drink",), []),
    "beer": (("drink",), []),
    "cocktail": (("drink",), []),
    "drink": ((), ["beverage"]),

    # ---- vehicles & transport ----
    "vehicle": ((), ["transport", "transportation"]),
    "car": (("vehicle",), ["automobile"]),
    "truck": (("vehicle",), []),
    "bus": (("vehicle",), []),
    "train": (("vehicle",), ["railway", "railroad"]),
    "bicycle": (("vehicle",), ["bike"]),
    "motorcycle": (("vehicle",), ["motorbike"]),
    "airplane": (("vehicle",), ["plane", "aircraft"]),
    "boat": (("vehicle",), ["ship", "sailboat"]),

    # ---- places & scenes ----
    "nature": ((), ["outdoors"]),
    "landscape": (("nature",), ["scenery", "view"]),
    "beach": (("nature",), ["seaside", "shore", "coast"]),
    "ocean": (("water", "nature"), ["sea"]),
    "lake": (("water", "nature"), []),
    "river": (("water", "nature"), ["stream"]),
    "waterfall": (("water", "nature"), []),
    "water": ((), []),
    "mountain": (("nature", "landscape"), ["mountains", "peak", "hill"]),
    "snow": (("winter",), []),
    "desert": (("nature", "landscape"), []),
    "field": (("nature", "landscape"), ["meadow"]),
    "park": (("nature",), []),
    "sky": ((), []),
    "cloud": (("sky",), []),
    "sunset": (("sky",), ["sunrise", "dusk", "dawn"]),
    "night sky": (("sky", "night"), ["stars"]),
    "night": ((), []),
    "rainbow": (("sky",), []),
    "city": ((), ["urban", "cityscape", "skyline"]),
    "street": (("city",), ["road"]),
    "building": (("city",), ["architecture"]),
    "house": (("building",), ["home"]),
    "bridge": (("city",), []),
    "tower": (("building",), []),
    "church": (("building",), ["cathedral", "temple"]),
    "castle": (("building",), ["palace"]),
    "interior": ((), ["indoors", "room"]),
    "kitchen": (("interior",), []),
    "bedroom": (("interior",), []),
    "living room": (("interior",), []),
    "office": (("interior",), []),
    "restaurant": (("interior",), ["cafe"]),

    # ---- seasons & events ----
    "spring": ((), []),
    "summer": ((), []),
    "autumn": ((), ["fall"]),
    "winter": ((), []),
    "event": ((), []),
    "birthday": (("event",), ["birthday party"]),
    "christmas": (("event",), ["christmas tree"]),
    "concert": (("event",), ["live music"]),
    "sports": (("event",), ["sport", "game"]),
    "fireworks": (("event", "night"), []),

    # ---- things ----
    "art": ((), ["artwork"]),
    "painting": (("art",), []),
    "drawing": (("art",), ["sketch", "illustration"]),
    "sculpture": (("art",), ["statue"]),
    "screenshot": ((), ["screen capture"]),
    "document": ((), ["paper", "receipt"]),
    "text": ((), ["sign", "writing"]),
    "book": ((), []),
    "computer": (("electronics",), ["laptop"]),
    "phone": (("electronics",), ["smartphone"]),
    "electronics": ((), []),
    "toy": ((), []),
    "clothing": ((), ["clothes", "fashion", "outfit"]),
    "shoe": (("clothing",), ["shoes", "sneakers"]),
    "jewelry": ((), ["necklace", "ring"]),
    "furniture": ((), []),
    "chair": (("furniture",), []),
    "table": (("furniture",), []),
    "musical instrument": ((), ["guitar", "piano"]),
    "ball": ((), []),
}


def ancestors(label, _seen=None):
    """All parents of a label, transitively (excluding the label itself)."""
    seen = _seen if _seen is not None else set()
    for parent in TAXONOMY.get(label, ((), []))[0]:
        if parent not in seen:
            seen.add(parent)
            ancestors(parent, seen)
    return seen


def _check():
    for label, (parents, _aliases) in TAXONOMY.items():
        for p in parents:
            assert p in TAXONOMY, f"{label!r} has unknown parent {p!r}"


_check()
