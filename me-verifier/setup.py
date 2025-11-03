import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from logger import setup_logger
from api.init import setup_manager

logger = setup_logger("setup")


def main():
    try:
        success = setup_manager.run_setup(skip_evaluation=False)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Configuración interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error no capturado: {e}")
        setup_manager.print_summary()
        sys.exit(1)


if __name__ == '__main__':
    main()