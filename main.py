from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainningPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainningPipeline
from mlProject.pipeline.stage_03_data_transformation import DatatransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import  ModelTrainingPipeline    

from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "nihadachir"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3d57c736a2992cee13f7a4bac3591b2c73123d95"



STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainningPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e    

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainningPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")      
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DatatransformationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")      
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainning Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = ModelTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")      
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"
try:    
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")              
except Exception as e:
    logger.exception(e)
    raise e    

