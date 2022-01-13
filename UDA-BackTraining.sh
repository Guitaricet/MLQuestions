set -e

echo "1. QG model generates questions from 50k passages and prepares synthetic data for back-training IR"
cd QG/
python generate.py --checkpoint NQ-checkpoint/ --file ../data/passages_unaligned.tsv

echo "2. Generated data is converted to json format for training IR model"
cd ../IR/
python gen_dpr_data.py --input_file ../QG/NQ-checkpoint/QG-predictions-50K.tsv --out_file outputs/BT.json

echo "3. IR model generates embeddings for 50k passages"
python generate_dense_embeddings.py --model_file NQ-checkpoint/bert-base-encoder.cp --ctx_file ../data/passages_unaligned.tsv --out_file NQ-checkpoint/embeddings_50k

echo "4. IR model retrieves passages from questions and prepares synthetic data for back-training QG"
python generate.py --model_file NQ-checkpoint/bert-base-encoder.cp --embeddings NQ-checkpoint/embeddings_50k_0.pkl --out_file ../QG/outputs/BT.tsv

echo "5. Train QG model on synthetic back-training data"
cd ../QG/
python train.py --epochs 5 --train_file outputs/BT.tsv --checkpoint NQ-checkpoint/

echo "6. Train IR model on synthetic back-training data"
cd ../IR/
python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --train_file outputs/BT.json --num_train_epochs 6 --model_file NQ-checkpoint/bert-base-encoder.cp --output_dir outputs/ --batch_size 32 --dev_file ../data/dev.json

echo "7. Evaluate QG model on test data"
cd ../QG/
python eval.py --checkpoint outputs/ --eval_file ../data/test.tsv

echo "8. Evaluate IR model on test data"
echo "a. Generate embeddings of 11k test passages"
cd ../IR/
python generate_dense_embeddings.py --model_file outputs/dpr_biencoder.5.1581 --ctx_file ../data/test_passages.tsv --out_file outputs/embeddings_11k
echo "b. evaluate top-k retrieval accuracy on test data"
python eval_retriever.py --model_file outputs/dpr_biencoder.5.1581 --embeddings outputs/embeddings_11k_0.pkl --eval_file ../data/test.tsv
