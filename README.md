## 設定
パスを通します
.bashrcに書き加えて置くと良い
```
export CUDA_MPS_PIPE_DIRECTORY=/path/to/dir
export CUDA_MPS_LOG_DIRECTORY=/path/to/dir
```
MPSデーモンを起動
```
nvidia-cuda-mps-control -d
```
MPSデーモンを確認
```
ps -ef | grep nvidia-cuda-mps-control
```
MPSデーモンを終了
```
echo quit | nvidia-cuda-mps-control
```
## 実行
```
nvcc RoopMemcpy.cu 
bash run.sh
```