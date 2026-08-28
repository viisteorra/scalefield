"""Short free-license motion clips from Wikimedia Commons. Small files only."""

from __future__ import annotations

# Direct upload.wikimedia.org URLs — no API round-trip (they 429 easily).
CLIPS: list[tuple[str, str]] = [
    ("clouds", "https://upload.wikimedia.org/wikipedia/commons/7/73/Clouds_transformation.webm"),
    ("sky_golden", "https://upload.wikimedia.org/wikipedia/commons/4/46/Sky_Golden.webm"),
    ("fountain", "https://upload.wikimedia.org/wikipedia/commons/d/d5/Fountain.ogv"),
    ("waves_iceland", "https://upload.wikimedia.org/wikipedia/commons/9/9b/Ocean_waves_at_L%C3%A6kjavik_beach%2C_Iceland.webm"),
    ("fireplace", "https://upload.wikimedia.org/wikipedia/commons/e/e8/Fire_burning.ogv"),
    ("rainfall", "https://upload.wikimedia.org/wikipedia/commons/e/e3/Rainfall_2.webm"),
    ("trees_wind", "https://upload.wikimedia.org/wikipedia/commons/b/b9/Trees_in_the_wind.webm"),
    ("waterfall", "https://upload.wikimedia.org/wikipedia/commons/2/27/Side_view_video_of_Kawaida_Waterfall_cascading%2C_Cianda%2C_Kiambu_County.webm"),
    ("sunset", "https://upload.wikimedia.org/wikipedia/commons/8/84/Golden_hour_sunset_at_Rumuigbo_Rivers_State.webm"),
    ("himachal", "https://upload.wikimedia.org/wikipedia/commons/c/ca/Natural_landscape_%28Himachal_Pradesh%29.webm"),
    ("mudpot", "https://upload.wikimedia.org/wikipedia/commons/1/13/Mudpot_at_Lassen_Volcanic_National_Park_in_August_2019.webm"),
    ("squirrel", "https://upload.wikimedia.org/wikipedia/commons/5/50/Squirrel_in_Dubrava_%C5%A0%C4%8Domyslickaja_natural_monument_%28Belarus%29_1.webm"),
    ("joshua", "https://upload.wikimedia.org/wikipedia/commons/d/d9/Joshua_Tree_National_Park_-_27327156848.webm"),
    ("stream", "https://upload.wikimedia.org/wikipedia/commons/e/ef/Running_Stream_Of_Water.webm"),
    ("grassland", "https://upload.wikimedia.org/wikipedia/commons/b/bc/Stream_and_grassland_habitat_in_Uganda_2026.webm"),
    ("flames", "https://upload.wikimedia.org/wikipedia/commons/8/89/Fire_flames_9652_Nevit.ogv"),
]
