import libsvm.*;

import java.io.*;
import java.nio.charset.StandardCharsets;

class SvmTrain {
    private SvmParameter param;        //通过命令行获取的各种参数
    private SvmProblem prob;        //读取的特征集和标签集数据转换为问题
    private String input_file_name; //输入文件路径
    private String model_file_name; //模型文件路径
    private int cross_validation;    //交叉验证选择核函数标志位，默认为0，可以通过-v 参数设置为1
    private int nr_fold;            //n重交叉验证

    private String separator = "\t";    //数据分隔符

    private static SvmPrintInterface svm_print_null = s -> {
    };

    private static void exit_with_help() {
        System.out.print("Usage: svm_train [options] training_set_file [model_file]\n"
                + "options:\n"

                //设置SVM的类型
                + "-s svm_type : set type of SVM (default 0)\n"
                + "	0 -- C-SVC		(multi-class classification)\n"
                + "	1 -- nu-SVC		(multi-class classification)\n"
                + "	2 -- one-class SVM\n"
                + "	3 -- epsilon-SVR	(regression)\n"
                + "	4 -- nu-SVR		(regression)\n"

                //设置核函数
                + "-t kernel_type : set type of kernel function (default 2)\n"
                + "	0 -- linear: u'*v\n"
                + "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
                + "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
                + "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
                + "	4 -- precomputed kernel (kernel values in training_set_file)\n"

                //设置degree，默认为3
                + "-d degree : set degree in kernel function (default 3)\n"

                //设置核函数中γ的值，默认为1/k，k为特征（或者说是属性）数
                + "-g gamma : set gamma in kernel function (default 1/num_features)\n"

                //设置核函数中的coef 0，默认值为0
                + "-r coef0 : set coef0 in kernel function (default 0)\n"

                //设置C-SVC、ε-SVR、n - SVR中从惩罚系数C，默认值为1
                + "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"

                //设置v-SVC、one-class-SVM与n-SVR中参数n，默认值0.5
                + "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"

                //设置v-SVR的损失函数中的e，默认值为0.1
                + "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"

                //设置cache内存大小，以MB为单位，默认值为100
                + "-m cachesize : set cache memory size in MB (default 100)\n"

                //设置终止准则中的可容忍偏差，默认值为0.001
                + "-e epsilon : set tolerance of termination criterion (default 0.001)\n"

                //是否使用启发式，可选值为0或1，默认值为1
                + "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"

                //是否计算SVC或SVR的概率估计，可选值0或1，默认0
                + "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"

                //对各类样本的惩罚系数C加权，默认值为1
                + "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"

                //n折交叉验证模式
                + "-v n : n-fold cross validation mode\n"

                //设置是否打印输出，默认有打印输出
                + "-q : quiet mode (no outputs)\n");
        System.exit(1);
    }

    //交叉验证选择最好的核函数
    private void doCrossValidation() {
        int total_correct = 0;  //分类正确的数量
        double total_error = 0; //分类错误的数量
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[prob.size];    //目标标签列表

        SVM.svmCrossValidation(prob, param, nr_fold, target);
        if (param.svm_type == SvmParameter.EPSILON_SVR || param.svm_type == SvmParameter.NU_SVR) {
            for (int i = 0; i < prob.size; i++) {
                double y = prob.label[i];
                double v = target[i];
                total_error += (v - y) * (v - y);
                sumv += v;
                sumy += y;
                sumvv += v * v;
                sumyy += y * y;
                sumvy += v * y;
            }
            System.out.print("Cross Validation Mean squared error = " + total_error / prob.size + "\n");
            System.out.print("Cross Validation Squared correlation coefficient = "
                    + ((prob.size * sumvy - sumv * sumy) * (prob.size * sumvy - sumv * sumy))
                    / ((prob.size * sumvv - sumv * sumv) * (prob.size * sumyy - sumy * sumy))
                    + "\n");
        } else {
            for (int i = 0; i < prob.size; i++)
                if (target[i] == prob.label[i])
                    ++total_correct;
            System.out.print("Cross Validation Accuracy = " + 100.0 * total_correct / prob.size + "%\n");
        }
    }

    private void run(String[] params) throws IOException {
        //解析命令行，将数据读入param，并获取input file、model file(可选)
        //读取命令行的文件名和参数，初始化svm_parameter结构，通过传入的参数或者默认值，设置一些必须的参数
        parse_command_line(params);

        //读取数据文件，并分配数据内存
        //初始化svm_problem结构
        //	- l：样本的行数
        //	- y：样本的所属类标签向量，lx1维
        //	- x：样本的所有特征向量，lxn维，其中n是样本的特征数
        read_problem();
        //参数检查
        String errorMsg = SVM.svmCheckParameter(prob, param);

        if (errorMsg != null) {
            System.err.print("ERROR: " + errorMsg + "\n");
            System.exit(1);
        }

        if (cross_validation != 0) {
            //根据交叉验证选择最好的核函数
            doCrossValidation();
        } else {
            //训练模型，填充svm_model结构
            //SVM模型
            SvmModel model = SVM.svm_train(prob, param);
            //保存模型
            SVM.svm_save_model(model_file_name, model);
        }
    }


    public static void main(String[] argv) throws IOException {

        String inputParams = "data/testSet.txt";
        String[] params = inputParams.split(" ");

        SvmTrain t = new SvmTrain();
        t.run(params);
    }

    private static double atof(String s) {
        double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            System.err.print("NaN or Infinity in input\n");
            System.exit(1);
        }
        return (d);
    }

    private static int atoi(String s) {
        return Integer.parseInt(s);
    }

    private void parse_command_line(String[] argv) {
        int i;
        SvmPrintInterface print_func = null; // default printing to stdout

        param = new SvmParameter();
        // default values
        param.svm_type = SvmParameter.C_SVC;
        param.kernel_type = SvmParameter.RBF;
        param.degree = 3;
        param.gamma = 0; // 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];
        cross_validation = 0;

        // parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-')
                break;
            if (++i >= argv.length)
                exit_with_help();
            switch (argv[i - 1].charAt(1)) {
                case 's':
                    param.svm_type = atoi(argv[i]);
                    break;
                case 't':
                    param.kernel_type = atoi(argv[i]);
                    break;
                case 'd':
                    param.degree = atoi(argv[i]);
                    break;
                case 'g':
                    param.gamma = atof(argv[i]);
                    break;
                case 'r':
                    param.coef0 = atof(argv[i]);
                    break;
                case 'n':
                    param.nu = atof(argv[i]);
                    break;
                case 'm':
                    param.cache_size = atof(argv[i]);
                    break;
                case 'c':
                    param.C = atof(argv[i]);
                    break;
                case 'e':
                    param.eps = atof(argv[i]);
                    break;
                case 'p':
                    param.p = atof(argv[i]);
                    break;
                case 'h':
                    param.shrinking = atoi(argv[i]);
                    break;
                case 'b':
                    param.probability = atoi(argv[i]);
                    break;
                case 'q':
                    print_func = svm_print_null;
                    i--;
                    break;
                case 'v':
                    cross_validation = 1;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        System.err.print("n-fold cross validation: n must >= 2\n");
                        exit_with_help();
                    }
                    break;
                case 'w':
                    ++param.nr_weight;
                {
                    int[] old = param.weight_label;
                    param.weight_label = new int[param.nr_weight];
                    System.arraycopy(old, 0, param.weight_label, 0, param.nr_weight - 1);
                }

                {
                    double[] old = param.weight;
                    param.weight = new double[param.nr_weight];
                    System.arraycopy(old, 0, param.weight, 0, param.nr_weight - 1);
                }

                param.weight_label[param.nr_weight - 1] = atoi(argv[i - 1].substring(2));
                param.weight[param.nr_weight - 1] = atof(argv[i]);
                break;
                default:
                    System.err.print("Unknown option: " + argv[i - 1] + "\n");
                    exit_with_help();
            }
        }

        //设置打印函数
        SVM.svm_set_print_string_function(print_func);

        if (i >= argv.length)
            exit_with_help();

        //检测输入文件路径
        input_file_name = argv[i];

        if (i < argv.length - 1)
            //如果输入文件路径之后还有参数的话，就是model文件路径
            model_file_name = argv[i + 1];
        else {
            //在输入文件路径中获取输入文件名
            int p = argv[i].lastIndexOf('/') + 1;
            model_file_name = argv[i].substring(p) + ".model";
        }
    }

    // read in a problem (in svmlight format)

    private void read_problem() throws IOException {

    	//获取输入文件的数据，存放至内存中
        try (BufferedReader in =
                     new BufferedReader(
                             new InputStreamReader(
                                     new FileInputStream(input_file_name), StandardCharsets.UTF_8), 512 * 1024)) {
            String line;
            while ((line = in.readLine()) != null) {
                String[] fields = line.split(separator);
                Double[] tmp = new Double[fields.length - 1];
				for (int i = 0; i < fields.length - 1; i++) {
					tmp[i] = atof(fields[i]);
				}
                prob.X.add(tmp);
				prob.label.add(atof(fields[fields.length - 1]));
            }
			prob.size = prob.X.size();
        }

        //prob.y 记录每个样本的所属类别
        //prob.x 每个x存储一行数据
//        BufferedReader fp = new BufferedReader(new FileReader(input_file_name));
//        Vector<Double> vy = new Vector<>();
//        Vector<SvmNode[]> vx = new Vector<>();
//        int max_index = 0;    //存储最大的特征索引

//        while (true) {
//            //循环直至文件读完
//            String line = fp.readLine();
//            if (line == null)
//                break;
//            //切分数据
//            String[] st = line.split("\t");
//            //将数据存储到prob.x里面去
//            SvmNode[] x = new SvmNode[st.length - 1];
//            for (int i = 0; i < st.length - 1; i++) {
//                x[i] = new SvmNode();
//                x[i].index = i + 1;
//                x[i].value = atof(st[i]);
//            }
//            max_index = Math.max(st.length - 1, max_index);
//            //x存储索引和特征值
//            vx.addElement(x);
//            //y存储标签
//            vy.addElement(atof(st[st.length - 1]));
//        }

        //将数据集大小、数据集特征向量、数据集标签向量存储到prob里面去
//        prob = new SvmProblem();
//        prob.size = vy.size();
//        prob.X = new SvmNode[prob.size][];
//        for (int i = 0; i < prob.size; i++)
//            prob.X[i] = vx.elementAt(i);
//        prob.label = new double[prob.size];
//        for (int i = 0; i < prob.size; i++)
//            prob.label[i] = vy.elementAt(i);

        //计算一个默认的gamma值
//        if (param.gamma == 0 && max_index > 0)
//            param.gamma = 1.0 / max_index;
		if (param.gamma == 0 && prob.size > 0)
			param.gamma = 1.0 / prob.size;

        //如果是用户自定义的核函数
//        if (param.kernel_type == SvmParameter.PRECOMPUTED)
//            for (int i = 0; i < prob.size; i++) {
//                //如果特征集向量矩阵的索引值不为0
//                if (prob.X[i][0].index != 0) {
//                    System.err.print("Wrong kernel matrix: first column must be 0: sample_serial_number\n");
//                    System.exit(1);
//                }
//                if ((int) prob.X[i][0].value <= 0 || (int) prob.X[i][0].value > max_index) {
//                    System.err.print("Wrong input format: sample_serial_number out of range\n");
//                    System.exit(1);
//                }
//            }
//        fp.close();
    }
}
