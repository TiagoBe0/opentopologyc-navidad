from config.extractor_config import ExtractorConfig 

from gui.gui_extractor import ExtractorGUI
from core.pipeline import ExtractorPipeline
from config.extractor_config import ExtractorConfig




def main():
    gui = ExtractorGUI()
    
    gui.run()
    # Cuando ya existe config_extractor.json:
    cfg = ExtractorConfig.load_json("config_extractor.json")

    pipeline = ExtractorPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()



    