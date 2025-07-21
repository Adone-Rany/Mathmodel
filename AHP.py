import numpy as np
#-----------1.权重计算函数（均值法）----------- 
def calculate_weights(matrix):
    matrix=np.array(matrix,dtype=float) 
    col_sum=np.sum(matrix,axis=0)#列和 
    norm_matrix=matrix/col_sum#归一化矩阵 
    weights=np.mean(norm_matrix,axis=1)#每行均值作为权重 
    return weights,norm_matrix
#-----------2.一致性检验函数----------- 
def consistency_check(matrix,weights):
    n=matrix.shape[0]
    lambda_max=np.sum(np.dot(matrix,weights)/weights)/n#最大特征值 
    CI=(lambda_max-n)/(n-1)
    RI_dict={1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49} 
    RI=RI_dict[n] 
    CR=CI/RI if RI!=0 else 0 
    return lambda_max,CI,CR
#-----------3.综合权重计算函数----------- 
def final_weights(criteria_weights,alternative_weights_matrix): 
    return np.dot(criteria_weights,alternative_weights_matrix)
#-----------4.数据输入部分-----------

#准则层成对比较矩阵 
criteria_matrix=[ 
[1,3,0.5], 
[1/3,1,0.25], 
[2,4,1]
]
#方案层针对每个准则的权重（归一化结果，已知） 
alternative_weights_matrix=np.array([ 
[0.30,0.25,0.10],#针对友好程度 
[0.55,0.60,0.30],#针对活跃程度 
[0.15,0.15,0.60]#针对护卫能力 
])
#-----------5.计算准则层权重-----------
criteria_weights,norm_criteria_matrix=calculate_weights(criteria_matrix) 
print("准则层权重:",criteria_weights)

#-----------6.一致性检验-----------
lambda_max,CI,CR=consistency_check(np.array(criteria_matrix),criteria_weights) 
print("一致性检验->最大特征值:{:.4f},CI:{:.4f},CR:{:.4f}".format(lambda_max,CI,CR)) 
if CR<0.1:
    print("一致性检验通过！") 
else:
    print("一致性检验未通过，请调整矩阵！")

#-----------7.计算最终综合权重-----------
final_result=final_weights(criteria_weights,alternative_weights_matrix) 
print("最终方案权重:",final_result)

#-----------8.排序输出-----------
dogs=["拉布拉多（A1）","边牧（A2）","德国牧羊犬（A3）"] 
ranking=sorted(zip(dogs,final_result),key=lambda x:x[1],reverse=True) 
print("推荐排序：")
for i,(dog,weight)in enumerate(ranking,1):
    print(f"{i}.{dog}:{weight:.3f}")