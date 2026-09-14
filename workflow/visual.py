mlflow ui --backend-store-uri sqlite:////home/fei/workspace/mlflow.db
运行
python /home/fei/workspace/qlib/examples/workflow_by_code.py


mlflow migrate-filestore \
  --backend-store-uri sqlite:////home/fei/workspace/mlflow.db \
  --backend-store-uri-file-store /home/fei/workspace/mlruns

mlflow ui --backend-store-uri sqlite:////home/fei/workspace/mlflow.db
