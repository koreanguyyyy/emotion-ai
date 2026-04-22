# tools/__init__.py
from tools.music_tool import recommend_music
from tools.book_tool import recommend_book
from tools.place_tool import recommend_place

ALL_TOOLS = [recommend_music, recommend_book, recommend_place]
