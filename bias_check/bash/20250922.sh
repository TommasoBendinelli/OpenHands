idenfitier=cross_check_experiment
models=("anthropic/claude-sonnet-4" "google/gemini-2.5-pro" "openai/gpt-5")
export OPENROUTER_API_KEY="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"
for examinee_model in "${models[@]}"; do
  echo "examinee_model: $examinee_model"
  for generator_model in "${models[@]}"; do
    echo python3 bias_check/generate_stories_w_questions.py num_questions=5 num_stories=20 generator_model="$generator_model" examinee_model="$examinee_model" idenfitier="$idenfitier"
    python3 bias_check/generate_stories_w_questions.py num_questions=5 num_stories=20 generator_model="$generator_model" examinee_model="$examinee_model" idenfitier="$idenfitier"
  done
done
