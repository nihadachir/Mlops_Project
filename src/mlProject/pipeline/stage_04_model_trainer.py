from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_trainer import ModelTrainer
from mlProject import logger
from pathlib import Path




STAGE_NAME = "Model Trainning stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                model_trainer_config = config.get_model_trainer_config()
                model_trainer_config = ModelTrainer(config=model_trainer_config)
                model_trainer_config.train()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)





if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e






