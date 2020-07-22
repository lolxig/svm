//
// svm_model
//
package libsvm;
public class SvmModel implements java.io.Serializable
{
	public SvmParameter param;	//保存-s -t -d -g -r等参数
	public int nr_class;		//分类的个数
	public int l;				//样本总数
	public SvmNode[][] SV;		//支持向量
	public double[][] sv_coef;	//支持向量的系数
	public double[] rho;		//b = -rho
	public double[] probA;      // pariwise probability information
	public double[] probB;
	public int[] sv_indices;    //指示支持向量在训练集中的位置

	// for classification only
	public int[] label;			//每一个类的标签
	public int[] nSV;			//每一个类的支持向量个数
								//nSV[0] + nSV[1] + ... + nSV[k-1] = l
}
