instance=01_log_solution_within_1000
#04_specific_log_file
ticket_file=28041,27738,27979,28133,28268,28271,28340
#UseCase_EncoderError,UseCase_ErrorSummary,UseCase_ShopOrder_Summary,UseCase_ShopOrderLoadingTime,UseCase_SpecificValue

python main.py \
  api.base_url=http://10.55.64.55:30000/v1 \
  request.model=openai/swiss-ai/Apertus-8B-Instruct-2509 \
  request.instance_type=maxon \
  request.instance=$instance \
  request.ticket_file=$ticket_file \
  -m
