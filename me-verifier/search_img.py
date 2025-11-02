import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from bing_image_downloader import downloader
from logger import setup_logger

logger = setup_logger("descargar_fotos")

QUERY = ""
LIMIT = 100


def downloaderImg(limit, query, output_dir):
    try:
        downloader.download(
            query,
            limit=limit,
            output_dir=output_dir, 
            adult_filter_off=False,
            force_replace=False,
            timeout=60,
            verbose=True
        )
        logger.info("[Descarga completada]: Imágenes guardadas en: %s", output_dir)
    except Exception as e:
        logger.exception("Ha ocurrido un error durante la descarga: %s", e)


def definePath():
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'not_me')
    return base_dir

if __name__ == '__main__':
    output_dir = definePath()
    os.makedirs(output_dir, exist_ok=True)
   
    logger.info("Iniciando descarga: '%s' (límite=%d)", QUERY, LIMIT)
        
    downloaderImg(LIMIT, QUERY, output_dir)