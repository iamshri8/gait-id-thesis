import tempfile
import mlflow
import mlflow.keras
import json


class MLFlowLogger():
    """
    Class that logs the parameters and the artifacts to the mlflow methods.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def log_params(self):
        mlflow.log_param("num_of_val_ids", self.cfg['num_of_val_ids'])
        mlflow.log_param("val_ids", self.cfg['val_ids'])
        mlflow.log_param("window_width", self.cfg['window_width'])
        mlflow.log_param("vec_dim", self.cfg['vec_dim'])
        mlflow.log_param("epochs", self.cfg["epochs"])
        mlflow.log_param("learning_rate", self.cfg['learning_rate'])
        mlflow.log_param("mining", self.cfg['mining'])
        mlflow.log_param("batch_size", self.cfg["batch_size"])

    def log_tags(self, run_id):
        mlflow.set_tag("run id", run_id)

    def log_train_metrics(self, train_history, callbacks):
        for i in range(len(train_history.history['loss'])):
            mlflow.log_metric("train loss", train_history.history['loss'][i], step=i)
            mlflow.log_metric("val loss", train_history.history['val_loss'][i], step=i)
        
        train_r1_list, train_mAP_list, val_r1_list, val_mAP_list = callbacks.get_all_metrics()
        
        for index, r1 in enumerate(train_r1_list):
            mlflow.log_metric("train rank1 acc", r1, step=index+1)
        mlflow.log_metric("train best rank1 acc", max(train_r1_list))
        mlflow.log_metric("train best epoch", train_r1_list.index(max(train_r1_list))+1)
        for index, mAP in enumerate(train_mAP_list):
            mlflow.log_metric("train mAP acc", mAP, step=index+1)        
        
        for index, r1 in enumerate(val_r1_list):
            mlflow.log_metric("val rank1 acc", r1, step=index+1)
        mlflow.log_metric("val best rank1 acc", max(val_r1_list))
        mlflow.log_metric("val best epoch", val_r1_list.index(max(val_r1_list))+1)
        for index, mAP in enumerate(val_mAP_list):
            mlflow.log_metric("val mAP acc", mAP, step=index+1)
            

    def log_model(self, model):
        mlflow.keras.log_model(model, "models")

