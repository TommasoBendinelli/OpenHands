models=("anthropic/claude-sonnet-4" "google/gemini-2.5-pro" "openai/gpt-5")

for examinee_model in "${models[@]}"; do
  echo "examinee_model: $examinee_model"
  for generator_model in "${models[@]}"; do
    echo "generator_model: $generator_model"
    python3 bias_check/generate_stories_w_questions.py num_questions=5 num_stories=20 generator_model="$generator_model" examinee_model="$examinee_model"
  done
done
