"""
Ingredient taxonomies and verified ingredient associations for Food-101 and Vireo-172 classes.
Derived from Recipe1M+ recipe ingredient lists and Vireo Food-172 ingredient vocabularies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

# Curated canonical ingredients for Food-101 dishes (derived from Recipe1M+ common ingredients)
FOOD101_INGREDIENTS: dict[str, list[str]] = {
    "apple_pie": ["apples", "flour", "butter", "sugar", "cinnamon", "pie crust"],
    "baby_back_ribs": ["pork ribs", "barbecue sauce", "brown sugar", "paprika", "garlic powder"],
    "baklava": ["phyllo dough", "walnuts", "pistachios", "butter", "honey", "cinnamon"],
    "beef_carpaccio": ["beef tenderloin", "olive oil", "lemon juice", "parmesan cheese", "arugula", "capers"],
    "beef_tartare": ["raw beef", "egg yolk", "shallots", "capers", "dijon mustard", "worcestershire sauce"],
    "beet_salad": ["beets", "goat cheese", "walnuts", "arugula", "olive oil", "balsamic vinegar"],
    "beignets": ["flour", "yeast", "milk", "sugar", "powdered sugar", "butter"],
    "bibimbap": ["steamed rice", "beef", "spinach", "bean sprouts", "carrots", "egg", "gochujang"],
    "bread_pudding": ["stale bread", "milk", "eggs", "sugar", "vanilla", "butter", "raisins"],
    "breakfast_burrito": ["flour tortilla", "eggs", "cheese", "potatoes", "bacon", "salsa"],
    "bruschetta": ["baguette", "tomatoes", "basil", "garlic", "olive oil", "balsamic vinegar"],
    "caesar_salad": ["romaine lettuce", "parmesan cheese", "croutons", "caesar dressing", "lemon juice", "black pepper"],
    "cannoli": ["pastry shells", "ricotta cheese", "powdered sugar", "chocolate chips", "vanilla"],
    "caprese_salad": ["fresh mozzarella", "tomatoes", "fresh basil", "olive oil", "balsamic glaze", "salt"],
    "carrot_cake": ["carrots", "flour", "sugar", "eggs", "cinnamon", "cream cheese frosting"],
    "ceviche": ["raw fish", "lime juice", "red onion", "cilantro", "jalapeno", "salt"],
    "cheesecake": ["cream cheese", "graham cracker crust", "sugar", "eggs", "sour cream", "vanilla"],
    "cheese_plate": ["assorted cheeses", "grapes", "nuts", "crackers", "fig jam"],
    "chicken_curry": ["chicken", "curry powder", "onion", "garlic", "ginger", "coconut milk", "tomatoes"],
    "chicken_quesadilla": ["flour tortilla", "shredded chicken", "monterey jack cheese", "cheddar cheese", "salsa"],
    "chicken_wings": ["chicken wings", "hot sauce", "butter", "garlic powder", "blue cheese dressing"],
    "chocolate_cake": ["flour", "cocoa powder", "sugar", "eggs", "butter", "baking powder", "milk"],
    "chocolate_mousse": ["dark chocolate", "eggs", "sugar", "heavy cream", "vanilla extract"],
    "churros": ["flour", "water", "sugar", "cinnamon", "butter", "vegetable oil"],
    "clam_chowder": ["clams", "potatoes", "onions", "celery", "heavy cream", "butter", "bacon"],
    "club_sandwich": ["sliced bread", "sliced turkey", "bacon", "lettuce", "tomato", "mayonnaise"],
    "crab_cakes": ["crab meat", "breadcrumbs", "mayonnaise", "egg", "dijon mustard", "old bay seasoning"],
    "creme_brulee": ["heavy cream", "egg yolks", "sugar", "vanilla bean"],
    "croque_madame": ["bread", "ham", "gruyere cheese", "bechamel sauce", "butter", "fried egg"],
    "cup_cakes": ["flour", "sugar", "butter", "eggs", "milk", "vanilla", "buttercream frosting"],
    "deviled_eggs": ["hard-boiled eggs", "mayonnaise", "dijon mustard", "paprika", "vinegar"],
    "donuts": ["flour", "sugar", "yeast", "milk", "butter", "eggs", "sugar glaze"],
    "dumplings": ["dumpling wrappers", "ground pork", "cabbage", "green onions", "ginger", "soy sauce"],
    "edamame": ["soybean pods", "coarse sea salt", "water"],
    "eggs_benedict": ["english muffins", "poached eggs", "canadian bacon", "hollandaise sauce", "butter"],
    "escargots": ["snails", "butter", "garlic", "parsley", "white wine"],
    "falafel": ["chickpeas", "parsley", "cilantro", "garlic", "cumin", "coriander", "flour"],
    "filet_mignon": ["beef tenderloin", "butter", "rosemary", "garlic", "black pepper", "salt"],
    "fish_and_chips": ["white fish fillets", "flour", "beer batter", "potatoes", "tartar sauce"],
    "foie_gras": ["duck liver", "butter", "cognac", "salt", "black pepper", "brioche"],
    "french_fries": ["potatoes", "vegetable oil", "salt"],
    "french_onion_soup": ["yellow onions", "beef broth", "butter", "gruyere cheese", "baguette croutons"],
    "french_toast": ["sliced bread", "eggs", "milk", "cinnamon", "vanilla", "maple syrup", "butter"],
    "fried_calamari": ["calamari rings", "flour", "cornstarch", "lemon wedges", "marinara sauce"],
    "fried_rice": ["cooked jasmine rice", "eggs", "peas", "carrots", "green onions", "soy sauce", "sesame oil"],
    "frozen_yogurt": ["yogurt", "milk", "sugar", "berries"],
    "garlic_bread": ["baguette", "garlic", "butter", "parsley", "parmesan cheese"],
    "gnocchi": ["potatoes", "flour", "egg", "salt", "marinara sauce"],
    "greek_salad": ["cucumbers", "tomatoes", "kalamata olives", "feta cheese", "red onion", "oregano", "olive oil"],
    "grilled_cheese_sandwich": ["sliced bread", "cheddar cheese", "butter"],
    "grilled_salmon": ["salmon fillet", "olive oil", "lemon juice", "garlic", "dill", "salt", "black pepper"],
    "guacamole": ["avocados", "lime juice", "cilantro", "red onion", "roma tomatoes", "salt"],
    "gyoza": ["gyoza wrappers", "ground pork", "napa cabbage", "scallions", "ginger", "garlic", "sesame oil"],
    "hamburger": ["ground beef", "burger bun", "lettuce", "tomato", "cheddar cheese", "onion", "pickles"],
    "hot_and_sour_soup": ["chicken broth", "tofu", "mushrooms", "bamboo shoots", "egg", "vinegar", "white pepper"],
    "hot_dog": ["frankfurter", "hot dog bun", "mustard", "ketchup", "relish"],
    "huevos_rancheros": ["corn tortillas", "fried eggs", "ranchero sauce", "refried beans", "queso fresco", "avocado"],
    "hummus": ["chickpeas", "tahini", "lemon juice", "garlic", "olive oil", "cumin", "salt"],
    "ice_cream": ["cream", "milk", "sugar", "egg yolks", "vanilla extract"],
    "lasagna": ["lasagna noodles", "ground beef", "ricotta cheese", "mozzarella", "tomato sauce", "parmesan"],
    "lobster_bisque": ["lobster meat", "lobster stock", "heavy cream", "butter", "shallots", "brandy", "tomato paste"],
    "lobster_roll_sandwich": ["lobster meat", "split-top hot dog bun", "mayonnaise", "melted butter", "lemon juice", "celery"],
    "macaroni_and_cheese": ["elbow macaroni", "cheddar cheese", "milk", "butter", "flour"],
    "macarons": ["almond flour", "powdered sugar", "egg whites", "granulated sugar", "buttercream filling"],
    "miso_soup": ["dashi stock", "miso paste", "silken tofu", "wakame seaweed", "green onions"],
    "mussels": ["mussels", "white wine", "garlic", "shallots", "butter", "parsley"],
    "nachos": ["tortilla chips", "melted cheese", "jalapenos", "black beans", "salsa", "sour cream"],
    "omelette": ["eggs", "butter", "milk", "salt", "black pepper", "cheese"],
    "onion_rings": ["yellow onions", "flour", "cornmeal", "buttermilk", "oil for frying"],
    "oysters": ["raw oysters", "lemon wedges", "cocktail sauce", "mignonette sauce"],
    "pad_thai": ["rice noodles", "tofu", "shrimp", "eggs", "bean sprouts", "peanuts", "tamarind paste", "fish sauce"],
    "paella": ["bomba rice", "saffron", "chicken", "shrimp", "mussels", "bell peppers", "peas", "olive oil"],
    "pancakes": ["flour", "milk", "eggs", "baking powder", "sugar", "butter", "maple syrup"],
    "panna_cotta": ["heavy cream", "milk", "sugar", "gelatin", "vanilla", "berry coulis"],
    "peking_duck": ["whole duck", "maltose syrup", "five-spice powder", "cucumber", "scallions", "hoisin sauce", "pancakes"],
    "pho": ["beef broth", "rice noodles", "beef slices", "star anise", "cinnamon", "bean sprouts", "fresh basil", "lime"],
    "pizza": ["pizza dough", "tomato sauce", "mozzarella cheese", "olive oil", "basil"],
    "pork_chop": ["pork chops", "garlic", "rosemary", "olive oil", "butter", "black pepper"],
    "poutine": ["french fries", "cheese curds", "brown gravy"],
    "prime_rib": ["bone-in ribeye roast", "garlic", "rosemary", "thyme", "coarse salt", "black pepper"],
    "pulled_pork_sandwich": ["slow-cooked pork shoulder", "barbecue sauce", "coleslaw", "hamburger bun"],
    "ramen": ["ramen noodles", "pork broth", "chashu pork", "soft-boiled egg", "green onions", "nori", "menma"],
    "ravioli": ["pasta dough", "ricotta cheese", "spinach", "egg", "parmesan", "marinara sauce"],
    "red_velvet_cake": ["flour", "cocoa powder", "buttermilk", "butter", "sugar", "red food coloring", "cream cheese frosting"],
    "risotto": ["arborio rice", "chicken broth", "white wine", "parmesan cheese", "shallots", "butter"],
    "samosa": ["flour", "potatoes", "green peas", "cumin", "garam masala", "ginger", "green chili", "oil"],
    "sashimi": ["raw tuna", "raw salmon", "soy sauce", "wasabi", "pickled ginger"],
    "scallops": ["sea scallops", "butter", "garlic", "lemon juice", "parsley"],
    "seaweed_salad": ["wakame seaweed", "sesame oil", "rice vinegar", "soy sauce", "sugar", "sesame seeds"],
    "spaghetti_bolognese": ["spaghetti", "ground beef", "canned tomatoes", "onion", "garlic", "carrots", "olive oil", "parmesan"],
    "spaghetti_carbonara": ["spaghetti", "eggs", "pancetta", "pecorino romano", "black pepper"],
    "spring_rolls": ["rice paper wrappers", "cabbage", "carrots", "mushrooms", "glass noodles", "soy sauce"],
    "steak": ["beef steak", "butter", "garlic", "rosemary", "salt", "black pepper"],
    "strawberry_shortcake": ["strawberries", "shortcake biscuits", "sugar", "whipped cream"],
    "sushi": ["sushi rice", "nori", "raw fish", "cucumber", "avocado", "rice vinegar"],
    "tacos": ["corn tortillas", "ground beef or carne asada", "onion", "cilantro", "lime", "salsa"],
    "takoyaki": ["octopus pieces", "dashi batter", "takoyaki sauce", "japanese mayonnaise", "bonito flakes", "aonori"],
    "tiramisu": ["ladyfingers", "mascarpone cheese", "espresso", "egg yolks", "sugar", "cocoa powder"],
    "tuna_tartare": ["fresh ahi tuna", "avocado", "sesame oil", "soy sauce", "lime juice", "chives"],
    "waffles": ["flour", "milk", "eggs", "butter", "sugar", "baking powder", "maple syrup"],
}

# Curated canonical ingredients for Vireo Food-172 dishes (sample Asian dishes)
VIREO172_INGREDIENTS: dict[str, list[str]] = {
    "kung_pao_chicken": ["chicken", "peanuts", "dried red chilies", "sichuan peppercorn", "scallions", "soy sauce"],
    "mapo_tofu": ["tofu", "ground beef or pork", "doubanjiang", "sichuan peppercorn", "garlic", "scallions"],
    "sweet_and_sour_pork": ["pork loin", "pineapple", "bell peppers", "vinegar", "sugar", "ketchup"],
    "hot_pot": ["sliced beef", "napa cabbage", "mushrooms", "tofu", "fish balls", "spicy broth"],
    "dim_sum": ["shrimp", "pork", "wheat starch", "bamboo shoots", "sesame oil"],
    "congee": ["rice", "water", "ginger", "scallions", "century egg", "pork"],
    "dan_dan_noodles": ["egg noodles", "minced pork", "sui mi ya cai", "chili oil", "sesame paste", "sichuan pepper"],
    "peking_duck": ["duck", "scallions", "cucumber", "sweet bean sauce", "pancakes"],
    "chow_mein": ["egg noodles", "cabbage", "carrots", "bean sprouts", "soy sauce", "oyster sauce"],
    "egg_fried_rice": ["cooked rice", "eggs", "green onions", "soy sauce", "oil"],
    "wonton_soup": ["wonton wrappers", "ground pork", "shrimp", "chicken broth", "green onions", "bok choy"],
}


def get_canonical_ingredients(class_name: str) -> list[str]:
    """
    Look up canonical ingredients for a food class across Food-101 and Vireo-172.
    Returns empty list if unknown (never fabricated).
    """
    norm = class_name.lower().strip().replace(" ", "_").replace("-", "_")
    if norm in FOOD101_INGREDIENTS:
        return list(FOOD101_INGREDIENTS[norm])
    if norm in VIREO172_INGREDIENTS:
        return list(VIREO172_INGREDIENTS[norm])
    return []
