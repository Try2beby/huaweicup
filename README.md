> 华为杯 B24102460013 全部附件

以第一问 2ap 情形为例

- 依次运行 `./q1/main_step1_2ap.ipynb` 和 `./q1/main_step2_2ap.ipynb`, 得到 `./q1/df_3ap_1.csv` 和 `./q1/df_2ap_final.csv`，后者为最终所用训练数据特征；
- 在 `./q1/` 路径下，运行`python ./preprocess_1_2ap.py` 处理测试数据，得到`./q1/df_1_2ap_test_final.csv`；
- 在 `./q1/` 路径下，运行`python ./modeling_2ap.py` 对每种方法进行超参搜索，结果输出在`./q1/results/`中；
- `./q1/compare_and_plot_2ap.ipynb` 用来
  - （1）从上述搜参结果中取出K折平均R2最大的模型及对应超参，并用该模型预测 `seq_time`，输出至`./q1/results/y_1_2ap_test_pred.csv`；
  - （2）绘制图像，输出至`./q1/fig/`。
