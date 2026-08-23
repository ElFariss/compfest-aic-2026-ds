from app.config import get_settings
from app.server import serve


if __name__ == "__main__":
    serve(get_settings())
