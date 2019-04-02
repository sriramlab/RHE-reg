/** 
 All of this code is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
#include "helper.h"
#include "storage.h"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
//Intermediate Variables
int blocksize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr covariate;  
genotype g;
MatrixXdr geno_matrix; //(p,n)

int MAX_ITER;
int k,p,n;
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  


options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=false;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 


std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}
void read_cov(int Nind, std::string filename, std::string covname){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;  
	while(in>>b)
	{
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line);
		string temp; 
		for(int k=0; k<covNum; k++){
			in>>temp;
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue;  
			} 
			int cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				missing[k].push_back(j); 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur; 
				covariate(j,k) = cur; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur;
					cov_sum[k] = cov_sum[k]+cur; 
				}
		} 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (covNum - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
}
void read_pheno2(int Nind, std::string filename){
//	pheno.resize(Nind,1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 
//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID")
			phenocount++; 
	} 
	pheno.resize(Nind, phenocount);
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		double temp;
		//fid,iid
		//todo: fid iid mapping; 
		//todo: handle missing phenotype
		in>>temp; in>>temp; 
		for(int j=0; j<phenocount;j++) {
			in>>temp; 
			pheno(i,j)=temp; 
		}
		i++;
	}
	//cout<<pheno; 
}
void read_pheno(int Nind, std::string filename){
	pheno.resize(Nind, 1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	int i=0;  
	while(std::getline(ifs, line)){
		pheno(i,0) = atof(line.c_str());
		if(pheno(i,0)==-1)
			cout<<"WARNING: missing phenotype"<<endl; 
		i++;  
	}

}
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
	
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif


	//TODO: Memory Effecient SSE FastMultipy

	for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
		int p_base = seg_iter*g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
			for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
				res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
		}
	}

	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);		
	int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
	for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
			res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	for(int p_iter=0;p_iter<p;p_iter++){
 		for(int k_iter=0;k_iter<Ncol_op;k_iter++){		 
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));		
		}		
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif
	
	int Ncol_op = Nrows_op;

	//cout << "ncol_op = " << Ncol_op << endl;

	int seg_iter;
	for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
	}
	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

	for(int n_iter=0; n_iter<n; n_iter++)  {
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
			res(k_iter,n_iter) = y_e[n_iter][k_iter];
			y_e[n_iter][k_iter] = 0;
		}
	}
	
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset (sums_elements, 0, Nrows_op * sizeof(int));

 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		double sum_to_calc=0.0;		
 		for(int p_iter=0;p_iter<p;p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize));
			res(k_iter,n_iter)=temp;
		}
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	res = geno_matrix * op;
}

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	res = op * geno_matrix;
}

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}

pair<double,double> get_error_norm(MatrixXdr &c){
	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true);
	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l,d_l,v_l; 
	if(fast_mode)
        u_l = b_svd.matrixU();
    else
        u_l = Q * b_svd.matrixU();
	v_l = b_svd.matrixV();
	d_l = MatrixXdr::Zero(k,k);
	for(int kk=0;kk<k; kk++)
		d_l(kk,kk) = (b_svd.singularValues())(kk);
	
	MatrixXdr u_k,v_k,d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);
	d_k = MatrixXdr::Zero(k_orig,k_orig);
	for(int kk =0 ; kk < k_orig ; kk++)
		d_k(kk,kk)  =(b_svd.singularValues())(kk);

	MatrixXdr b_l,b_k;
    b_l = u_l * d_l * (v_l.transpose());
    b_k = u_k * d_k * (v_k.transpose());

    if(fast_mode){
        double temp_k = b_k.cwiseProduct(b).sum();
        double temp_l = b_l.cwiseProduct(b).sum();
        double b_knorm = b_k.norm();
        double b_lnorm = b_l.norm();
        double norm_k = (b_knorm*b_knorm) - (2*temp_k);
        double norm_l = (b_lnorm*b_lnorm) - (2*temp_l);	
        return make_pair(norm_k,norm_l);
    }
    else{
        MatrixXdr e_l(p,n);
        MatrixXdr e_k(p,n);
        for(int p_iter=0;p_iter<p;p_iter++){
            for(int n_iter=0;n_iter<n;n_iter++){
                e_l(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize) - b_l(p_iter,n_iter);
                e_k(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize) - b_k(p_iter,n_iter);
            }
        }

        double ek_norm = e_k.norm();
        double el_norm = e_l.norm();
        return make_pair(ek_norm,el_norm);
    }
}




void print_vals(){

	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true);
	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l; 
	u_l = b_svd.matrixU();
	MatrixXdr v_l;
	v_l = b_svd.matrixV();
	MatrixXdr u_k;
	MatrixXdr v_k,d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);


	if(debug){
		ofstream c_file;
		c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals.txt")).c_str());
		c_file<<c<<endl;
		c_file.close();
		
		d_k = MatrixXdr::Zero(k_orig,k_orig);
		for(int kk =0 ; kk < k_orig ; kk++)
			d_k(kk,kk)  =(b_svd.singularValues())(kk);
		MatrixXdr x_k;
		x_k = d_k * (v_k.transpose());
		ofstream x_file;
		x_file.open((string(command_line_opts.OUTPUT_PATH) + string("xvals.txt")).c_str());
		x_file<<x_k.transpose()<<endl;
		x_file.close();
	}
}

int main(int argc, char const *argv[]){

	clock_t io_begin = clock();
    clock_gettime (CLOCK_REALTIME, &t0);

	pair<double,double> prev_error = make_pair(0.0,0.0);
	double prevnll=0.0;

	parse_args(argc,argv);

	
	//TODO: Memory effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
	fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;

	
	if(text_version){
		if(fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH,missing);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH,missing);
	}
	else{
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH,missing,fast_mode);
	}

	//TODO: Implement these codes.
	if(missing && !fast_mode){
		cout<<"Missing version works only with mailman i.e. fast mode\n EXITING..."<<endl;
		exit(-1);
	}
	if(fast_mode && memory_efficient){
		cout<<"Memory effecient version for mailman EM not yet implemented"<<endl;
		cout<<"Ignoring Memory effecient Flag"<<endl;
	}
	if(missing && var_normalize){
		cout<<"Missing version works only without variance normalization\n EXITING..."<<endl;
		exit(-1);
	}

    //MAX_ITER =  command_line_opts.max_iterations ; 
	int B = command_line_opts.batchNum; 
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = false; 
	accelerated_em = command_line_opts.accelerated_em;
	k = k_orig + command_line_opts.l;
	k = (int)ceil(k/10.0)*10;
	command_line_opts.l = k - k_orig;
	p = g.Nsnp;
	n = g.Nindv;
	bool toStop=false;
		toStop=true;
	srand((unsigned int) time(0));
	c.resize(p,k);
	x.resize(k,n);
	v.resize(p,k);
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize);
	}
	
	clock_t io_end = clock();

	//TODO: Initialization of c with gaussian distribution
	c = MatrixXdr::Random(p,k);


	// Initial intermediate data structures
	blocksize = k;
	int hsegsize = g.segment_size_hori; 	// = log_3(n)
	int hsize = pow(3,hsegsize);		 
	int vsegsize = g.segment_size_ver; 		// = log_3(p)
	int vsize = pow(3,vsegsize);		 

	partialsums = new double [blocksize];
	sum_op = new double[blocksize];
	yint_e = new double [hsize*blocksize];
	yint_m = new double [hsize*blocksize];
	memset (yint_m, 0, hsize*blocksize * sizeof(double));
	memset (yint_e, 0, hsize*blocksize * sizeof(double));

	y_e  = new double*[g.Nindv];
	for (int i = 0 ; i < g.Nindv ; i++) {
		y_e[i] = new double[blocksize];
		memset (y_e[i], 0, blocksize * sizeof(double));
	}

	y_m = new double*[hsegsize];
	for (int i = 0 ; i < hsegsize ; i++)
		y_m[i] = new double[blocksize];

	for(int i=0;i<p;i++){
		means(i,0) = g.get_col_mean(i);
		stds(i,0) =1/ g.get_col_std(i);
		sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}



	ofstream c_file;
	if(debug){
		c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals_orig.txt")).c_str());
		c_file<<c<<endl;
		c_file.close();
		printf("Read Matrix\n");
	}

	cout<<"Running on Dataset of "<<g.Nsnp<<" SNPs and "<<g.Nindv<<" Individuals"<<endl;

	#if SSE_SUPPORT==1
		if(fast_mode)
			cout<<"Using Optimized SSE FastMultiply"<<endl;
	#endif


	//get geno
	//cout<<g.get_geno(0,0,false);
	//read phenotype
	//
	//
	std::string filename=command_line_opts.PHENOTYPE_FILE_PATH; 
	read_pheno2(g.Nindv, filename);
	if(filename=="")
	{	
		cout<<"No Phenotype File Specified"<<endl;
		return 0 ; 
	}
	std::string covfile=command_line_opts.COVARIATE_FILE_PATH;
        std::string covname=command_line_opts.COVARIATE_NAME;  
	if(covfile!=""){
		use_cov=true; 
		read_cov(g.Nindv, covfile, covname); 
	} 
	else if(covfile=="")
		cout<<"No Covariate File Specified"<<endl; 
	double y_sum=pheno.sum(); 
	double y_mean = y_sum/g.Nindv;
	for(int i=0; i<g.Nindv; i++) 
		pheno(i,0) =pheno(i,0) - y_mean; //center phenotype	
	y_sum=pheno.sum(); 
	//correctness check 
	//MatrixXdr zb = MatrixXdr::Random(1, g.Nsnp); 
	//MatrixXdr res(1,g.Nindv); 
	//multiply_y_post_fast(zb, 1, res, false); 
	//cout<< MatrixXdr::Constant(1,4,1)<<endl;  
	MatrixXdr res(g.Nsnp, 1); 
	multiply_y_pre_fast(pheno,1,res,false); 
	res = res.cwiseProduct(stds); 
	MatrixXdr resid(g.Nsnp, 1); 
	resid = means.cwiseProduct(stds); 
	resid = resid *y_sum; 	
	MatrixXdr Xy(g.Nsnp,1); 
	Xy = res-resid; 
	double yKy = (Xy.array()* Xy.array()).sum(); 
	yKy = yKy/g.Nsnp; 
	double tr_k =0 ;
	MatrixXdr temp = sum2 + g.Nindv* means.cwiseProduct(means) - 2 * means.cwiseProduct(sum);
	temp = temp.cwiseProduct(stds);
	temp = temp.cwiseProduct(stds); 
	tr_k = temp.sum() / g.Nsnp;
	double tr_k2=0;
	VectorXd vec(g.Nsnp);
	for(int i=0; i<g.Nsnp; i++)
		vec(i)=stds(i); 

	clock_t it_begin =clock(); 
	for(int i=0; i<B; i++){
		MatrixXdr zb= MatrixXdr::Random(g.Nindv, 10);
		zb = zb * sqrt(3); 
		MatrixXdr res(g.Nsnp, 10); 
		multiply_y_pre_fast(zb,10,res, false);
		MatrixXdr zb_sum = zb.colwise().sum(); 
		for(int j=0; j<g.Nsnp; j++)
		        for(int k=0; k<10;k++)
		             res(j,k) = res(j,k)*stds(j,0); 
		MatrixXdr resid(g.Nsnp, 10);
		MatrixXdr inter = means.cwiseProduct(stds);
		resid = inter * zb_sum;
		zb = res - resid; // X^Tzb =zb' 
		clock_t Xtzb = clock(); 
		
              	for(int k=0; k<10; k++){
                  for(int j=0; j<g.Nsnp;j++){
                        zb(j,k) =zb(j,k) *stds(j,0);}}
                                              
		MatrixXdr new_zb = zb.transpose(); 
		MatrixXdr new_res(10, g.Nindv);
		multiply_y_post_fast(new_zb, 10, new_res, false); 
		MatrixXdr new_resid(10, g.Nsnp); 
		MatrixXdr zb_scale_sum = new_zb * means;
		new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);  
		tr_k2+= ((new_res - new_resid).array() * (new_res-new_resid).array()).sum();  
		
	}
	tr_k2  = tr_k2 /10/g.Nsnp/g.Nsnp/B; 

	MatrixXdr A(2,2); 
	A(0,0)=tr_k2; A(0,1)=tr_k; A(1,0)=tr_k; A(1,1)= g.Nindv; 
	MatrixXdr b(2,1); 
	b(0,0) = yKy; 
	b(1,0) = (pheno.array() * pheno.array()).sum(); 
	MatrixXdr herit = A.colPivHouseholderQr().solve(b); 
	cout<<"V(G): "<<herit(0,0)<<endl; 
	cout<<"V(e): "<<herit(1,0)<<endl; 
	cout<<"Vp "<<herit.sum()<<endl; 
	cout<<"V(G)/Vp: "<<herit(0,0)/herit.sum()<<endl;   
	double c = g.Nindv* (tr_k2/g.Nindv - tr_k*tr_k/g.Nindv/g.Nindv); 
	cout<<"SE: "<<sqrt(2/c)<<endl; 
	clock_t it_end = clock();

    print_vals();
		
	clock_t total_end = clock();
	double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
	double avg_it_time = double(it_end - it_begin) / (B * 1.0 * CLOCKS_PER_SEC);
	double total_time = double(total_end - total_begin) / CLOCKS_PER_SEC;
	cout<<"IO Time:  "<< io_time << "\nAVG Iteration Time:  "<<avg_it_time<<"\nTotal runtime:   "<<total_time<<endl;

	delete[] sum_op;
	delete[] partialsums;
	delete[] yint_e; 
	delete[] yint_m;

	for (int i  = 0 ; i < hsegsize; i++)
		delete[] y_m [i]; 
	delete[] y_m;

	for (int i  = 0 ; i < g.Nindv; i++)
		delete[] y_e[i]; 
	delete[] y_e;

	return 0;
}
