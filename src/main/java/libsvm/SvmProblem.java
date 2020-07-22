package libsvm;

import java.util.List;

public class SvmProblem implements java.io.Serializable
{
	public int size;			//数据集的大小
	public List<Double> label;	//数据集标签向量
	public List<Double[]> X;	//数据集特征矩阵
}
