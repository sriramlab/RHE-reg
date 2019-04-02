/** 
 Mailman algorithm is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
 RHE_reg is written by Yue Ariel Wu
 (UCLA)
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

#include "boost/random.hpp"
#include "boost/accumulators/statistics/stats.hpp"
#include "boost/math/distributions/chi_squared.hpp"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions.hpp"

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

//clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr pheno_prime; 
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


//solve Ax= b, use for b
//
MatrixXdr yy; 
MatrixXdr yKy; 
MatrixXdr Xy; 
//use for covariate
MatrixXdr WW; 

//use for missing phenotype
MatrixXdr pheno_mask2; 
vector<int> pheno_mask; 

//use for genetic correlation
vector<vector<int> > gc_index; 
options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=true;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false;
bool compute_gc=false;  
bool reg = true;
bool gwas=false;
bool bpheno=false;   
bool pheno_fill=false;  // if pheno_fill, estimate tr[k2] once and use for all phenotypes 
bool noh2g=false; 
vector<string> pheno_name; 

std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}
int read_gc(std::string filename)
{
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 

	int count=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp1, temp2; 
		in>>temp1; in>>temp2; 
		gc_index.push_back(vector<int>()); 
		gc_index[count].push_back(atoi(temp1.c_str())); 
		gc_index[count].push_back(atoi(temp2.c_str()));
		count++;

	}
	for(int i=0; i<count; i++)
		cout<<gc_index[i][0]<<endl; 
	return count; 
}
int read_cov(bool std,int Nind, std::string filename, std::string covname){
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
		if(b!="FID" && b !="IID"){
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
		}
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
		in>>temp; in>>temp; //FID IID 
		for(int k=0; k<covNum; k++){
			
			in>>temp;
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue; 
			} 
			double cur = atof(temp.c_str()); 
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
		//if(j<10) 
		//	cout<<covariate.block(j,0,1, covNum)<<endl; 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (Nind - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
	if(std)
	{
		MatrixXdr cov_std;
		cov_std.resize(1,covNum);  
		MatrixXdr sum = covariate.colwise().sum();
		MatrixXdr sum2 = (covariate.cwiseProduct(covariate)).colwise().sum();
		MatrixXdr temp;
//		temp.resize(Nind, 1); 
//		for(int i=0; i<Nind; i++)
//			temp(i,0)=1;  
		for(int b=0; b<covNum; b++)
		{
			cov_std(0,b) = sum2(0,b) + Nind*cov_sum[b]*cov_sum[b]- 2*cov_sum[b]*sum(0,b);
			cov_std(0,b) =sqrt((Nind- 1)/cov_std(0,b)) ;
			double scalar=cov_std(0,b); 
			for(int j=0; j<Nind; j++)
			{
				covariate(j,b) = covariate(j,b)-cov_sum[b];  
				covariate(j,b) =covariate(j,b)*scalar;
			} 
			//covariate.col(b) = covariate.col(b) -temp*cov_sum[b];
			
		}
	}	
	return covNum; 
}
int read_pheno2(int Nind, std::string filename,int pheno_idx, bool pheno_fill){
//	pheno.resize(Nind,1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 
	vector<vector<int> > missing; 
//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
			phenocount++;
			missing.push_back(vector<int>());  
			pheno_name.push_back(b); 
		}
	}
	if(pheno_idx !=0)
		pheno_name[0] = pheno_name[pheno_idx-1];   
	vector<double> pheno_sum(phenocount,0); 
	if(pheno_idx !=0){
		pheno.resize(Nind,1);
		pheno_mask2.resize(Nind,1);
	} 
	else{
		pheno.resize(Nind, phenocount);
		pheno_mask2.resize(Nind, phenocount);
	} 
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		//todo: fid iid mapping; 
		in>>temp; in>>temp; 
		for(int j=0; j<phenocount;j++) {
			in>>temp;
			if(pheno_idx !=0 && j==(pheno_idx-1))
			{
				if(temp=="NA")
				{	
					missing[j].push_back(i); 
					pheno_mask.push_back(0); 
					pheno_mask2(i,0)=0; 
					pheno(i,0)=0; 
				}
				else
				{
					double cur = atof(temp.c_str()); 
					pheno(i,0)=cur; 
					pheno_mask.push_back(1); 
					pheno_mask2(i,0)=1;
					pheno_sum[j]=pheno_sum[j]+cur;  
				}
			}
			if(pheno_idx ==0){
				if(temp=="NA")
				{
					missing[j].push_back(i); 
					pheno_mask.push_back(0); 
					pheno_mask2(i,j)=0;
					pheno(i,j)=0; 
				} 
				else{
					double cur= atof(temp.c_str()); 
					pheno(i,j)=cur; 
					pheno_sum[j] = pheno_sum[j]+cur; 
					pheno_mask2(i,j)=1; 
				}
			}
		}
		i++;
	}
	//not performing phenotype imputation
	if(! pheno_fill){
		if(pheno_idx!=0)
			return 1; 
		return phenocount;
	} 
	//fill missing with mean
	for(int a=0; a<phenocount; a++)
	{
		int missing_num= missing[a].size(); 
		double	pheno_avg = pheno_sum[a]/(Nind- missing_num); 
		//cout<<"pheno "<<a<<" avg: "<<pheno_avg<<endl; 
		for(int b=0 ; b<missing_num; b++)
		{
			int index = missing[a][b]; 
			pheno(index, a)= pheno_avg; 
			pheno_mask2(index,a)=1;
		}
	}
	return phenocount; 
}

void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means, int exist_ind,int phenoindex){
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	double vg, ve; 
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
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter, phenoindex)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter,phenoindex,exist_ind));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means, int exist_ind, int phenoindex){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter,phenoindex,exist_ind));		
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
 			sum_to_calc += g.get_col_mean(p_iter,phenoindex)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res, int exist_ind, int phenoindex){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize,phenoindex, exist_ind)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,int exist_ind, int phenoindex){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize,phenoindex, exist_ind));
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

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means,int exist_ind, int phenoindex){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means,exist_ind,phenoindex);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res,exist_ind,phenoindex);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means,int exist_ind, int phenoindex){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means, exist_ind, phenoindex);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res,exist_ind, phenoindex);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}
pair<double,double> weightedjack(vector<double> &t, vector<double> &m, double theta){
	if(t.size() != m.size())
	{
		cerr<<"In functions::weightedjack, Mismatch in length of t and m" <<endl; 
		exit(1); 
	}
	int g=0; 
	//double n=vector::sum(m); 
	double n=0; 
	for(int i=0; i<m.size(); i++)
		n += m[i]; 
	vector <double> res(m.size(), 0); 
	double a=0; 
	for(int i=0; i<m.size(); i++)
	{
		if(m[i]<=0)
			continue; 
		g++; 
		double h = n/m[i]; 
		a += (1-1/h)*t[i]; 
		double r = theta * h - (h-1) * t[i]; 
		res[i] =r; 
	}
	if(g==0){
		cerr<<"In function::weightedjack. Nmber of bolcks == 0"<<endl; 
		exit(1); 
	}
	double tj = theta * g; 
	tj -=a ; 
	double sj =0; 
	for(int i=0; i<m.size(); i++)
	{	
		if(m[i]<=0)
			continue; 
		double h=n/m[i]; 
		sj += pow((res[i]-tj), 2)/(h-1); 
	}
	sj /= g; 
	sj = pow(sj, 0.5); 
	
	return pair<double, double> (tj,sj); 


}
double compute_jack_knife(int j, int k, double rg)
{
	MatrixXdr Xy1 = Xy.block(0, j, g.Nsnp, 1); 
	MatrixXdr Xy2 = Xy.block(0, k, g.Nsnp, 1);
	MatrixXdr y1  = pheno.block(0, j, g.Nindv, 1); 
	MatrixXdr y2 = pheno.block(0,k, g.Nindv, 1);  
	vector<double> jack_knife_rg(1000,1); 
	int block_len = g.Nsnp / 1000; 
	for(int i=0; i<1000; i++)
	{
		int len = block_len; 
		if(i*block_len+block_len > g.Nsnp)
			len = g.Nsnp - i*block_len; 
		MatrixXdr Xy1_cur  = Xy1.block(block_len*i, 0, len, 1); 
		MatrixXdr Xy2_cur = Xy2.block(block_len*i, 0, len,1); 

		
		double X =yKy(j,k)*g.Nsnp/(g.Nsnp-len) - yy(j,k);  
		X = X- (Xy1_cur.transpose()*Xy2_cur/(g.Nsnp-len)).sum(); 
			
		double  Y = yKy(j,j)*g.Nsnp/(g.Nsnp-len)- yy(j,j);  
		Y = Y-  (Xy1_cur.transpose()* Xy1_cur/(g.Nsnp-len)).sum(); 
		Y = sqrt(Y); 
		double Z = yKy(k,k)*g.Nsnp/(g.Nsnp-len)- yy(k,k); 
		Z = Z- (Xy2_cur.transpose()*Xy2_cur/(g.Nsnp-len)).sum(); 
		Z = sqrt(Z); 
		jack_knife_rg[i] = X / Y/Z; 
		/*if(i<10){
			cout<<jack_knife_rg(i,0)<<endl; 	
			cout<<"X: " << X <<endl <<"Y: "<<Y <<endl <<"Z: "<<Z <<endl; 
		}*/
	}
	pair<double, double> jack_output; 
	vector<double> jack_weight(1000,1); 
	//weighted jacknife SE
	jack_output = weightedjack(jack_knife_rg, jack_weight, rg); 

	cout<<"Weighted jackknife SE: "<<jack_output.second<<endl; 
	//unweighted
	double result = 0; 
	for(int i=0; i<1000; i++)
	{	
		double temp = jack_knife_rg[i]-rg; 
		result = result + temp*temp; 
	}
	//cout<<jack_knife_rg.block(0,0,10,1); 
	result = result *(1000-1) / 1000; 
	cout<<"Unweighted jackknife SE: "<<sqrt(result) <<endl; 
	return sqrt(result); 

}

void compute_se_coh(MatrixXdr &Xy1,MatrixXdr &Xy2, MatrixXdr &y1,MatrixXdr &y2,MatrixXdr &se, double h2g1,double h2g2, double h2e1,double h2e2,double tr_k2, int B , int exist_ind1, int exist_ind2, double rho_g, double rho_e, double h2g1_se, double h2g2_se, double h2g1_yKy, double h2g1_yy, double h2g2_yKy, double h2g2_yy)
{
	h2g1_se = h2g1_se*h2g1_se; 
	h2g2_se = h2g2_se* h2g2_se; 
//compute tr[y_1y_1^T(K-I)(h2g2K+h2e2I)(K-I)] 
//substude with tr[y_1y_1^T (K-I) y_2y_2^T (K-I)] 
	//compute X^Ty1, X^Ty2
	MatrixXdr zb=Xy1;
	MatrixXdr zb2 =Xy2; 
	//compute XXy1, XXy2
	 for(int j=0; j<p; j++){
                zb(j, 0)= zb(j,0)*stds(j,0);
		zb2(j,0) = zb2(j,0)*stds(j,0); 
	}
        MatrixXdr new_zb = zb.transpose();
	MatrixXdr new_zb2 = zb2.transpose();
        MatrixXdr new_res(1,n);
	MatrixXdr new_res2(1,n); 
        multiply_y_post_fast(new_zb, 1, new_res,false, exist_ind1,0);
	multiply_y_post_fast(new_zb2,1,new_res2, false, exist_ind2, 0); 
        MatrixXdr new_resid(1,p);
	MatrixXdr new_resid2(1,p); 
	MatrixXdr zb_scale_sum = new_zb*means;
	MatrixXdr zb_scale_sum2 = new_zb2 *means; 
        new_resid= zb_scale_sum* MatrixXdr::Constant(1,n, 1);
	new_resid2 = zb_scale_sum2*MatrixXdr::Constant(1,n,1); 
        MatrixXdr alpha1 = (new_res-new_resid).transpose() ;
	MatrixXdr alpha2 = (new_res2 -new_resid2).transpose(); 
        //compute Ky1, ky2
         for(int j=0; j< n; j++){
                alpha1(j,0)= alpha1(j,0)/p ;
		alpha2(j,0) = alpha2(j,0)/p; 
	}
	//alpha1 = (K-I)y1, alpha2 = (K-I)y2
	alpha1 = alpha1-y1;
	alpha2 = alpha2-y2;  
	MatrixXdr var_A = y2.transpose() * alpha1; 
	var_A(0,0) = var_A(0,0)*var_A(0,0); 
	//var_A(0,0) = tr[y1y1^T (K-I) y2y2^T(K-I)] 
	//compute tr[y1 y2^T(K-I)(rg*K+re*I)(K-I)] = rg(alpha2^T XX^T alpha1)/M + alpha2^Talpha1 re
	//compute X^Talpha1, X^T alpha2
	MatrixXdr res1(p,1); 
	MatrixXdr res2(p,1); 
	MatrixXdr resid1(p,1);
	MatrixXdr resid2(p,1); 
	MatrixXdr inter = means.cwiseProduct(stds); 
	multiply_y_pre_fast(alpha1, 1, res1, false, exist_ind1,0); 
	multiply_y_pre_fast(alpha2, 1, res2, false, exist_ind2,0); 
	for(int j=0; j<p; j++)
	{
		res1(j,0)= res1(j,0)*stds(j,0); 
		res2(j,0) = res2(j,0)*stds(j,0); 
	}
	resid1 = inter* alpha1.sum(); 
	resid2 = inter* alpha2.sum(); 
	MatrixXdr Xalpha1(p,1); 
	MatrixXdr Xalpha2(p,1); 
	Xalpha1 = res1 -resid1; 
	Xalpha2 = res2 -resid2; 
	//tr[y1 y2^T(K-I)(rg*K+re*I)(K-I)] = result
	double yKy1 = (Xalpha1.array()*Xalpha1.array()).sum()/p; 
	double temp1 = (alpha1.array()*alpha1.array()).sum();  
	double yKy = (Xalpha1.array()*Xalpha2.array()).sum()/p; 
	double temp = (alpha1.array()*alpha2.array()).sum(); 
	double result = yKy*rho_g + temp*rho_e + yKy1*h2g2+temp1*h2e2; 
	var_A(0,0)=0; 
	double var_X =  result; 
	double cor_XY = 2*(yKy* h2g1+ temp*h2e1) ; 
	double cor_XZ = 2*(yKy*h2g2 + temp*h2g2);
	double cor_YZ = 2*(yKy*rho_g + temp*rho_e);  

	double var_Y2 = 2*(yKy1*h2g1+ temp1*h2g1); 
	double yKy2 = (Xalpha2.array()*Xalpha2.array()).sum()/p; 
	double temp2 = (alpha2.array()*alpha2.array()).sum(); 
	double var_Z2 = 2*(yKy2*h2g2 + temp2*h2g2); 
	double E_X = tr_k2 *rho_g - exist_ind1 *rho_g; 
	double E_Y2 = tr_k2 *h2g1 - exist_ind1 *h2g1; 
	double E_Z2 = tr_k2 *h2g2 -exist_ind1*h2g2; 
	
	double E_Y = sqrt(E_Y2) - var_Y2/8/E_Y2/(sqrt(E_Y2)); 
	double E_Z = sqrt(E_Z2) - var_Z2 /8/E_Z2/(sqrt(E_Z2)); 
	double var_Y = var_Y2 / 4 / E_Y2; 
	double var_Z = var_Z2 / 4 / E_Z2; 
 
	double final_result = var_X / E_Y/E_Y /E_Z/E_Z  + var_Y*E_X*E_X / E_Y/E_Y/E_Y/E_Y/E_Z/E_Z + var_Z*E_X*E_X /E_Y /E_Y/E_Z/E_Z/E_Z/E_Z ; 

	se(0,0)=final_result;	
 
	//var_A(0,0) = var_A(0,0) + result; 
	//cout <<"var_A: "<<var_A(0,0)<<endl; 
	//double mu_A = rho_g * tr_k2 - exist_ind1* rho_g; 
	//cout<<"mu_A: "<<mu_A<<endl; 
	//double mu_B = tr_k2 - exist_ind1;
	//cout<<"mu_B: "<<mu_B<<endl;  
	//double var_B = tr_k2 / B /10;
	//cout<<"var B: " <<var_B<<endl ; 
	//double var_rg = var_A(0,0)/mu_B/mu_B + mu_A*mu_A*var_B/mu_B/mu_B/mu_B/mu_B; 
	//cout<<"var_rg: "<<var_rg<<endl; 
	//double factor = tr_k2 -2*exist_ind1 + exist_ind1*exist_ind1 / tr_k2; 
	//double E_rg = rho_g + rho_g/B/10/factor; 
	//cout<<"E_rg: " <<E_rg<<endl; 
	//double E_h2g1_2 = h2g1 + h2g1/B/10/factor;
	//cout<<"E_h2g1^2: "<<E_h2g1_2<<endl;  
	//double E_h2g2_2 = h2g2 + h2g2/B/10/factor;
	//cout<<"E_h2g2^2: "<<E_h2g2_2<<endl;  

	//double E_h2g1 = sqrt(E_h2g1_2) - h2g1_se / E_h2g1_2 / sqrt(E_h2g1_2) / 8; 
	//cout<<"E_h2g1: "<<E_h2g1<<endl; 
	//double E_h2g2 = sqrt(E_h2g2_2) - h2g2_se / E_h2g2_2 / sqrt(E_h2g2_2) / 8 ; 
	//cout<<"E_h2g2: "<<E_h2g2<<endl; 
	//double var_h2g1 = h2g1_se  / 4 / E_h2g1_2;
	//cout<<"var(h2g1): "<<var_h2g1<<endl;  
	//double var_h2g2 = h2g2_se / 4 / E_h2g2_2;
	//cout<<"var(h2g2): "<<var_h2g2<<endl;   
	//double final_result = var_rg / E_h2g1/E_h2g1/E_h2g2/E_h2g2 + E_rg* E_rg * var_h2g1 / E_h2g1/E_h2g1/E_h2g1/E_h2g1 /E_h2g2 /E_h2g2 + E_rg *E_rg * var_h2g2 /E_h2g1/E_h2g1/E_h2g2/E_h2g2/E_h2g2/E_h2g2; 
	//se(0,0)=final_result;  	
}

//previous version; computing se without covariate
void compute_se1(MatrixXdr &Xy,  MatrixXdr &y,MatrixXdr &se, double h2g, double h2e,double tr_k2, int B , int exist_ind)
{
	//compute tr[yy^T(K-I)(h2gK+h2eI)(K-I)]
	//compute X^T y 
	//imput X^y[i] p*1 vector
	cout<<"p: "<<p << "  n: "<<n<<endl;
        MatrixXdr zb=Xy;
	//compute XXy
	for(int j=0; j<p; j++)
                zb(j, 0)= zb(j,0)*stds(j,0);
        MatrixXdr new_zb = zb.transpose();
        MatrixXdr new_res(1,n);
        multiply_y_post_fast(new_zb, 1, new_res,false, exist_ind,0);
        MatrixXdr new_resid(1,p);
        MatrixXdr zb_scale_sum = new_zb*means;
        new_resid= zb_scale_sum* MatrixXdr::Constant(1,n, 1);
        MatrixXdr alpha = (new_res-new_resid).transpose() ;
	//compute Ky
	 for(int j=0; j< n; j++)
                alpha(j,0)= alpha(j,0)/p ;
	//alpha =(K-I)y
	 alpha = alpha -y;
        MatrixXdr res(p,1);
        MatrixXdr resid(p,1);
        MatrixXdr inter = means.cwiseProduct(stds);
        multiply_y_pre_fast(alpha, 1, res, false, exist_ind,0);
        for(int j=0; j<p;j++)
                res(j,0)=res(j,0)*stds(j,0);
        inter = means.cwiseProduct(stds);
        resid = inter * alpha.sum();
        MatrixXdr Xalpha(p,1);
        Xalpha = res-resid;
	//Xy =res; 
	double yKy = (Xalpha.array()*Xalpha.array()).sum() / p;
        double temp =(alpha.array()*alpha.array()).sum();
        double result = yKy*h2g+temp*h2e;
        result = 2*result + h2g*h2g*tr_k2/10/B;
	result = sqrt(result) / (tr_k2-n);
        cout<<result<<endl; 
   	MatrixXdr result1(1,1);
        result1(0,0)=result;se=result1;
}

void compute_se(MatrixXdr &Xy, MatrixXdr &y,MatrixXdr &se, double h2g, double h2e,double tr_k2, int B , int exist_ind, double tr_k, int cov_num)
{
	//mu_a and mu_B are for covarate use to compute scalar
	double mu_A = h2g * (tr_k2 - tr_k * tr_k); 
	double mu_B = tr_k2 - tr_k*tr_k / (exist_ind - cov_num); 
	//compute X^T y
	//input X^y[i] p*1 vector
	cout<<"p: "<<p << "  n: "<<n<<endl; 
	MatrixXdr zb=Xy;
	//compute XXy
	for(int j=0; j<p; j++)
		zb(j, 0)= zb(j,0)*stds(j,0); 
	MatrixXdr new_zb = zb.transpose(); 
	MatrixXdr new_res(1,n); 
	multiply_y_post_fast(new_zb, 1, new_res,false, exist_ind,0); 
	MatrixXdr new_resid(1,p); 
	MatrixXdr zb_scale_sum = new_zb*means; 
	new_resid= zb_scale_sum* MatrixXdr::Constant(1,n, 1); 
	MatrixXdr alpha = (new_res-new_resid).transpose() ;
	//compute Ky
	for(int j=0; j< n; j++)
		alpha(j,0)= alpha(j,0)/p ;
	//alpha = (K-I)y 
	//if no covariate, tr_k / (exist_ind - cov_num) = 1
	if(cov_num==0)
		alpha = alpha -((tr_k)/(exist_ind-cov_num))*y; 
	if(cov_num != 0){
		alpha = alpha - covariate * WW * (covariate.transpose()*alpha);
		alpha = alpha - ((tr_k)/(exist_ind-cov_num))*y; 
		MatrixXdr alpha_prime = covariate.transpose() * alpha; 
		alpha = alpha - covariate * WW * alpha_prime; 
	}
	MatrixXdr res(p,1); 
	MatrixXdr resid(p,1); 
	MatrixXdr inter = means.cwiseProduct(stds); 
	multiply_y_pre_fast(alpha, 1, res, false, exist_ind,0); 
	for(int j=0; j<p;j++)
		res(j,0)=res(j,0)*stds(j,0); 
	inter = means.cwiseProduct(stds); 
	resid = inter * alpha.sum(); 
	MatrixXdr Xalpha(p,1); 
	Xalpha = res-resid;
	//Xy =res;
	double yKy = (Xalpha.array()*Xalpha.array()).sum() / p; 
	double temp =(alpha.array()*alpha.array()).sum();  
	double result = yKy*h2g+temp*h2e; 
	if(cov_num==0)
		result = 2*result + h2g*h2g*tr_k2/10/B; 
	else	
		result = 2*result +tr_k2* mu_A*mu_A / mu_B/mu_B/10/B; 
	//cout<<result; 
	if(cov_num==0)
		result = sqrt(result) / (tr_k2-n);
	else 
		result = sqrt(result) / mu_B;   
	MatrixXdr result1(1,1); 
	result1(0,0)=result;se=result1;  
}
//compute_b2 for genetic correlation
void compute_b2(bool use_cov,  double exist_ind, int pheno_i, int pheno_j, MatrixXdr& pheno_prime1, MatrixXdr& pheno_prime2)
{
	int pheno_num=1; 
	MatrixXdr y1 = pheno.block(0,pheno_i, g.Nindv, 1); 
	MatrixXdr y2 = pheno.block(0,pheno_j, g.Nindv, 1); 
	MatrixXdr mask1 = pheno_mask2.block(0, pheno_i, g.Nindv, 1); 
	MatrixXdr mask2 = pheno_mask2.block(0, pheno_j, g.Nindv, 1); 
	MatrixXdr y_sum1 = y1.colwise().sum(); 
	MatrixXdr y_sum2 = y2.colwise().sum(); 
	if(!use_cov)
	{
		 MatrixXdr res1(g.Nsnp, pheno_num);
		MatrixXdr res2(g.Nsnp, pheno_num); 
                multiply_y_pre(y1,pheno_num,res1,false, exist_ind,0);
		multiply_y_pre(y2, pheno_num, res2, false, exist_ind,0); 
                for(int i=0; i<pheno_num; i++){
                        MatrixXdr cur= res1.block(0,i,g.Nsnp, 1);
                        res1.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                	cur  = res2.block(0,i,g.Nsnp, 1); 
			res2.block(0,i,g.Nsnp, 1) = cur.cwiseProduct(stds) ;
		}
		MatrixXdr resid1(g.Nsnp, pheno_num);
		MatrixXdr resid2(g.Nsnp, pheno_num); 
                for(int i=0; i<pheno_num; i++)
                {
                        resid1.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum1(0,i);
			resid2.block(0,i,g.Nsnp,1 ) =means.cwiseProduct(stds)*y_sum2(0,i); 
                }
		MatrixXdr Xy1 = res1-resid1; 
		MatrixXdr Xy2 = res2-resid2; 		
		yKy = Xy1.transpose() * Xy2; 
		yKy = yKy /g.Nsnp; 
	}
	if(use_cov)
	{
		MatrixXdr y_temp1 = y1 - covariate * WW *pheno_prime1; 
		MatrixXdr y_temp2 = y2-covariate*WW* pheno_prime2; 
		y_temp1 = y_temp1.cwiseProduct(mask1); 
		y_temp2 = y_temp2.cwiseProduct(mask2); 
		y_sum1 = y_temp1.colwise().sum(); 
		y_sum2 = y_temp2.colwise().sum(); 
			
		
		MatrixXdr res1(g.Nsnp, pheno_num); 
		MatrixXdr res2(g.Nsnp, pheno_num); 
		multiply_y_pre(y_temp1, pheno_num, res1, false, exist_ind,0); 
		multiply_y_pre(y_temp2, pheno_num, res2, false, exist_ind,0); 

		for(int i=0; i<pheno_num; i++){
                        MatrixXdr cur= res1.block(0,i,g.Nsnp, 1);
                        res1.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        cur  = res2.block(0,i,g.Nsnp, 1);
                        res2.block(0,i,g.Nsnp, 1) = cur.cwiseProduct(stds) ;
                }
                MatrixXdr resid1(g.Nsnp, pheno_num);
                MatrixXdr resid2(g.Nsnp, pheno_num);
                for(int i=0; i<pheno_num; i++)
                {
                        resid1.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum1(0,i);
                        resid2.block(0,i,g.Nsnp,1 ) =means.cwiseProduct(stds)*y_sum2(0,i);
                }
		MatrixXdr Xy1 = res1-resid1;
                MatrixXdr Xy2 = res2-resid2;
                yKy = Xy1.transpose() * Xy2;
                yKy = yKy /g.Nsnp;

	}
	yy= y1.transpose()* y2; 
	if(use_cov)
		yy = yy- pheno_prime1.transpose() * WW * pheno_prime2; 


}
//compute_b1 for heritability
void compute_b1 (bool use_cov, MatrixXdr& y_sum, double  exist_ind,int pheno_i , MatrixXdr& pheno_prime_cur, bool pheno_fill, int pheno_num){
//	double exist_ind = exist_ind_mx(0,0);
	MatrixXdr pheno_cur, mask_cur; 
	if (!pheno_fill) 
		{
		 pheno_cur = pheno.block(0, pheno_i, g.Nindv, 1); 
		 mask_cur = pheno_mask2.block(0,pheno_i, g.Nindv, 1); 
		}
	else 
	{
		pheno_cur = pheno; 
		mask_cur = pheno_mask2; 
	}
	if(!use_cov){
		 MatrixXdr res(g.Nsnp, pheno_num);
                multiply_y_pre(pheno_cur,pheno_num,res,false, exist_ind,0);
                for(int i=0; i<pheno_num; i++){
                        MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                        res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                }
                MatrixXdr resid(g.Nsnp, pheno_num);
                for(int i=0; i<pheno_num; i++)
                {
                        resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum(0,i);
                }
  		if(pheno_fill){
			Xy = res-resid; 
			yKy = Xy.transpose() * Xy;
                        yKy = yKy/g.Nsnp;        
                }
		else{
		Xy.block(0,pheno_i, g.Nsnp,1) = res-resid;
                MatrixXdr temp = (res-resid).transpose() * (res-resid); 
		yKy(pheno_i,pheno_i)  = temp(0,0); 
		yKy(pheno_i, pheno_i)  = yKy(pheno_i, pheno_i) / g.Nsnp; 		   }
	}
	if(use_cov)
	{
		  MatrixXdr y_temp = pheno_cur-covariate* WW * pheno_prime_cur;
                  y_temp = y_temp.cwiseProduct(mask_cur); 
                cout<<"y_temp: "<<y_temp.block(0,0,10,1)<<endl;
	 	MatrixXdr y_temp_sum=y_temp.colwise().sum();
	
		MatrixXdr res(g.Nsnp, pheno_num);
                        multiply_y_pre(y_temp,pheno_num,res,false,exist_ind,0);
                        for(int i=0; i<pheno_num; i++){
                                MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                                res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        }
                        MatrixXdr resid(g.Nsnp, pheno_num);
                          for(int i=0; i<pheno_num; i++)
                        {
                                resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_temp_sum(0,i);
                        }
			if(pheno_fill){
                        Xy = res-resid;
                        yKy = Xy.transpose() * Xy;
                        yKy = yKy/g.Nsnp;
                	}
                	else{
                	Xy.block(0,pheno_i, g.Nsnp,1) = res-resid;
                	MatrixXdr temp = (res-resid).transpose() * (res-resid);
                	yKy(pheno_i,pheno_i)  = temp(0,0);
                	yKy(pheno_i, pheno_i)  = yKy(pheno_i, pheno_i) / g.Nsnp;                   }
                      //  Xy = res-resid;
                      //  yKy = Xy.transpose() * Xy; 
		      //  yKy = yKy/g.Nsnp; 
	}
	yy = pheno_cur.transpose() * pheno_cur; 
	if(use_cov)
		yy =yy- pheno_prime_cur.transpose()* WW *pheno_prime_cur; 

}
//previous version: read only one phenotype, use vector for missing mask
//out dated
void compute_b(bool use_cov, int pheno_num, MatrixXdr &y_sum ,int exist_ind){
	if(!use_cov){
	 if(pheno_num<10)
        {
		MatrixXdr res(g.Nsnp, 1); 
//                MatrixXdr res(g.Nsnp, pheno_num);
                multiply_y_pre(pheno,pheno_num,res,false, exist_ind,0);
                for(int i=0; i<pheno_num; i++){
                        MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                        res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                }
                MatrixXdr resid(g.Nsnp, pheno_num);
                for(int i=0; i<pheno_num; i++)
                {
                        resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum(0,i);
                }
                Xy = res-resid;
                MatrixXdr temp = Xy.transpose() *Xy;
		yKy=temp.diagonal(); 
	}
	else
	{
		for(int i=0; i*10<pheno_num; i++){
                int col_num = 10;
                if( (pheno_num-i*10)<10)
                        col_num = pheno_num-i*10;
                MatrixXdr pheno_block = pheno.block( 0, i*10,g.Nindv, col_num);
                MatrixXdr res(g.Nsnp, col_num);
                multiply_y_pre(pheno_block,col_num,res,false, exist_ind,0);
                for(int j=0; j<col_num; j++){
                        MatrixXdr cur= res.block(0,j,g.Nsnp, 1);
                        res.block(0,j,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                }
                MatrixXdr resid(g.Nsnp, col_num);
                for(int j=0; j<col_num; j++)
                {
                        resid.block(0,j,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum(0,i*10+j);
                }
		Xy.block(0, i*10, g.Nsnp, col_num) = res-resid;
                MatrixXdr Xy_cur = Xy.block(0, i*10, g.Nsnp, col_num);
                MatrixXdr temp = Xy_cur.transpose() * Xy_cur;
                yKy.block(i*10, 0, col_num, 1)  = temp.diagonal();
	        }

	}
	yKy=yKy/g.Nsnp;
	} 
	if(use_cov)
        {
		MatrixXdr y_temp = pheno-covariate* WW * pheno_prime;
		for(int i=0; i<g.Nindv; i++)
			y_temp(i,0) = y_temp(i,0)*pheno_mask[i];
		cout<<"y_temp: "<<y_temp.block(0,0,10,1)<<endl;  
		//y_temp = y_temp.cwiseProcduct(pheno_mask); 
		MatrixXdr y_temp_sum=y_temp.colwise().sum();
                if(pheno_num<10)
                {
                        MatrixXdr res(g.Nsnp, pheno_num);
                        multiply_y_pre(y_temp,pheno_num,res,false,exist_ind,0);
                        for(int i=0; i<pheno_num; i++){
                                MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                                res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        }
                        MatrixXdr resid(g.Nsnp, pheno_num);
                          for(int i=0; i<pheno_num; i++)
                        {
                                resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_temp_sum(0,i);
                        }
                        Xy = res-resid;
                        MatrixXdr temp = Xy.transpose() *Xy;
                        yKy = temp.diagonal();


                }

		else{
                	for(int j=0; j*10<pheno_num; j++){
                        int col_num = 10;
                        if( (pheno_num-j*10)<10)
                                col_num = pheno_num-j*10;
                        MatrixXdr pheno_block = y_temp.block( 0, j*10,g.Nindv, col_num);
                        MatrixXdr res(g.Nsnp, col_num);

                        multiply_y_pre(pheno_block,col_num,res,false, exist_ind,0);
                        for(int i=0; i<col_num; i++){
                                MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                                res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        }
                        MatrixXdr resid(g.Nsnp, col_num);
                        for(int i=0; i<col_num; i++)
                        {
                                resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_temp_sum(0,j*10+i);
                        }
                        Xy.block(0, j*10, g.Nsnp, col_num) = res-resid;
                        MatrixXdr Xy_cur = Xy.block(0, j*10, g.Nsnp, col_num);
                        MatrixXdr temp = Xy_cur.transpose()*Xy_cur;
                        yKy.block(j*10, 0, col_num, 1)  = temp.diagonal();


                	}
                }
                yKy = yKy/g.Nsnp;
        }

	yy= pheno.transpose() * pheno;
        if(use_cov)
                yy= yy- pheno_prime.transpose() * WW * pheno_prime;


}
void rhe_reg(double &tr_k2, double &tr_k_rsid, int B, int pheno_i, int exist_ind)
{
	MatrixXdr cur_mask = pheno_mask2.block(0, pheno_i, g.Nindv, 1);
	for(int i=0; i<B; i++){
		 //G^T zb
		 //clock_t random_step=clock();
		MatrixXdr zb= MatrixXdr::Random(g.Nindv, 10);
                zb = zb * sqrt(3);
                for(int b=0; b<10; b++){
                        MatrixXdr temp = zb.block(0,b,g.Nindv,1);
                        zb.block(0,b,g.Nindv, 1) = temp.cwiseProduct(cur_mask);
                }
                MatrixXdr res(g.Nsnp, 10);
                multiply_y_pre(zb,10,res, false, exist_ind,pheno_i);
		//sigma scale \Sigma G^T zb; compute zb column sum
		 MatrixXdr zb_sum = zb.colwise().sum();
		//std::vector<double> zb_sum(10,0);
		//res = Sigma*res;
		 for(int j=0; j<g.Nsnp; j++)
                        for(int k=0; k<10;k++)
                             res(j,k) = res(j,k)*stds(j,0);
		//compute /Sigma_^T M z_b
		MatrixXdr resid(g.Nsnp, 10);
                MatrixXdr inter = means.cwiseProduct(stds);
                resid = inter * zb_sum;
                MatrixXdr zb1(g.Nindv,10);
                zb1 = res - resid; // X^Tzb =zb'

		//compute zb' %*% /Sigma
		//zb = Sigma * zb
		for(int k=0; k<10; k++){
                  for(int j=0; j<g.Nsnp;j++){
                        zb1(j,k) =zb1(j,k) *stds(j,0);}}

                MatrixXdr new_zb = zb1.transpose();
                MatrixXdr new_res(10, g.Nindv);
                multiply_y_post(new_zb, 10, new_res, false, exist_ind,pheno_i);

		//new_res =  zb' \Sigma G^T  10*N 
		MatrixXdr new_resid(10, g.Nindv);
                MatrixXdr zb_scale_sum = new_zb * means;
                new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);
                MatrixXdr Xzb = new_res- new_resid;
                for( int b=0; b<10; b++)
                {
                        MatrixXdr temp = Xzb.block(b,0,1, g.Nindv);
                        Xzb.block(b,0,1, g.Nindv) = temp.cwiseProduct(cur_mask.transpose());
                }
		if(use_cov)
                {
                        MatrixXdr temp1 = WW * covariate.transpose() *Xzb.transpose();
                        MatrixXdr temp = covariate * temp1;
                        MatrixXdr Wzb  = zb.transpose() * temp;
                        tr_k_rsid += Wzb.trace();

                        Xzb = Xzb - temp.transpose();
                }
                tr_k2+= (Xzb.array() * Xzb.array()).sum();
	}

	tr_k2  = tr_k2 /10/g.Nsnp/g.Nsnp/B;
	tr_k_rsid = tr_k_rsid/10/g.Nsnp/B;
}
void compute_A(bool use_cov, MatrixXdr& A, int B)
{

}
//solve for Ax=b efficiently, return A^{-1}b = x, where A is fixted to be sigma_g^2XX^T/M+sigma_eI 
void conjugate_gradient(int n, double vg, double ve,MatrixXdr &A,  MatrixXdr &b, MatrixXdr &x , int exist_ind){
        int k=0;
        double thres=0.0001;//1e-4
        int max_iter=50;
        MatrixXdr r0(n, 1);
        MatrixXdr r1(n, 1);
        MatrixXdr p(n, 1);
        MatrixXdr s(n, 1);
        for(int i=0; i<n; i++)
        {       x(i,0)=0;
		//p(i,0)=0; 
        }
        double temp=1;
        double beta,alpha;
        r0=b;
        r1=b;
        MatrixXdr mask_sum = A.colwise().sum();
        while(temp>thres && k<max_iter){
                k++;
                if(k==1)
                        p = b;
                else
                {
                        MatrixXdr temp1 = r0.transpose() * r0;
                        MatrixXdr temp2 = r1.transpose() * r1;
                        beta = temp2(0,0)/ temp1(0,0);
                        p = r1+ beta*p;
                }
		s=A*p ; 
		MatrixXdr temp1 = r1.transpose() * r1; 
		MatrixXdr temp2 = p.transpose()*s; 
		alpha = temp1(0,0)/ temp2(0,0); 
		x = x+alpha * p ; 
		MatrixXdr r2= r1; 
		r1 = r1 - alpha* s; 
		r0 = r2; 
		MatrixXdr z = r1.transpose() * r1; 
		temp = z(0,0); 
                //use mailman to compute s=Ap  = vg* XX^T p + ve * p  
                //                    s = A*p ; 
        	cout<<"Iter: "<< k <<"  " << temp <<endl; 
	}

}
void conjugate_gradient_mailman(int n, double vg, double ve,MatrixXdr &geno_mask,  MatrixXdr &b, MatrixXdr &x , int exist_ind,int cur_snp){
        vg = vg/g.Nsnp* cur_snp; 
	ve = 1-vg; 
	int k=0;
        double thres=0.0005;//1e-5
        int max_iter=50;
        MatrixXdr r0(n, 1);
        MatrixXdr r1(n, 1);
        MatrixXdr p(n, 1);
        MatrixXdr s(n, 1);
        for(int i=0; i<n; i++)
        {       x(i,0)=0;
        }
        double temp=1;
        double beta,alpha;
        r0=b;
        r1=b;
        while(temp>thres && k<max_iter){
                k++;
                if(k==1)
                        p = b;
                else
                {
                        MatrixXdr temp1 = r0.transpose() * r0;
                        MatrixXdr temp2 = r1.transpose() * r1;
                        beta = temp2(0,0)/ temp1(0,0);
                        p = r1+ beta*p;
                }
                //s=A*p ;
                MatrixXdr res(g.Nsnp,1); 
		multiply_y_pre(p,1, res, false, exist_ind,0); 
		MatrixXdr p_sum = p.colwise().sum(); 
		for(int j=0; j<g.Nsnp; j++)
			res(j,0) = res(j,0)*stds(j,0); 
		MatrixXdr resid(g.Nsnp, 1); 
		MatrixXdr inter = means.cwiseProduct(stds); 
		resid = inter * p_sum; 
		MatrixXdr zb1 = res -resid; 
		zb1 = zb1.cwiseProduct(geno_mask); 
		for(int j=0; j<g.Nsnp; j++)
			zb1(j,0) = zb1(j,0)*stds(j,0); 

		MatrixXdr new_zb = zb1.transpose(); 
		MatrixXdr new_res(1, g.Nindv); 
		multiply_y_post(new_zb, 1, new_res, false, exist_ind, 0); 
		MatrixXdr new_resid(1,g.Nindv) ;
		MatrixXdr zb_scale_sum = new_zb *means; 
		new_resid = zb_scale_sum* MatrixXdr::Constant(1,g.Nindv, 1); 
		MatrixXdr Xzb = new_res- new_resid; 
		s= vg*Xzb.transpose()/cur_snp + ve*p; 


                MatrixXdr temp1 = r1.transpose() * r1;
                MatrixXdr temp2 = p.transpose()*s;
                alpha = temp1(0,0)/ temp2(0,0);
                x = x+alpha * p ;
                MatrixXdr r2= r1;
                r1 = r1 - alpha* s;
                r0 = r2;
                MatrixXdr z = r1.transpose() * r1;
                temp = z(0,0);
		cout<<"Iter : "<<k <<" "<< temp <<endl; 
	}
}
void conjugate_gradient_mailman2(int n, double vg, double ve,MatrixXdr &geno_mask,  MatrixXdr &b, MatrixXdr &x , int exist_ind,int cur_snp, MatrixXdr &x_guess, MatrixXdr &Linv){
        vg = vg/g.Nsnp* cur_snp;
        ve = 1-vg;
        int iter=0;
        double thres=0.0005;//1e-5
	 int max_iter=50;
     	int colNum = b.cols(); 
	int rowNum = b.rows(); 
	MatrixXdr r0(rowNum, colNum);
        MatrixXdr r1(rowNum, colNum);
	MatrixXdr v1(rowNum, colNum); 
        MatrixXdr Av1(rowNum, colNum);
	x = x_guess;  
        double temp=1;
        double t,s;
	//compute Ax
	MatrixXdr Ax(n,colNum); 
	
	MatrixXdr res(g.Nsnp, colNum); 
	MatrixXdr Cx = x_guess;
	for (int k=0; k<colNum ;k ++)
                        Cx.block(0, k, g.Nindv, 1) = Cx.block(0,k,g.Nindv,1).cwiseProduct( Linv);
	multiply_y_pre(Cx, colNum ,res, false, exist_ind, 0); 
	MatrixXdr p_sum= Cx.colwise().sum(); 
	for(int j=0; j<g.Nsnp; j++)
		for (int k=0; k<colNum; k++)
			res(j,k) = res(j,k)*stds(j,0); 
	MatrixXdr resid(g.Nsnp, colNum); 
	MatrixXdr inter= means.cwiseProduct(stds); 
	resid  = inter* p_sum; 
	MatrixXdr zb1 = res - resid; 
	for(int k=0; k<colNum; k++)
		zb1.block(0,k,g.Nsnp, 1)=zb1.block(0,k,g.Nsnp, 1).cwiseProduct(geno_mask); 
	for(int j=0; j<g.Nsnp; j++) 
		for(int k=0; k<colNum; k++)
		zb1(j,k) = zb1(j,k)*stds(j,0); 
	MatrixXdr new_zb = zb1.transpose(); 
	MatrixXdr new_res(colNum,g.Nindv); 
	multiply_y_post(new_zb, colNum, new_res, false, exist_ind, 0); 
	MatrixXdr new_resid(colNum, g.Nindv); 
	MatrixXdr zb_scale_sum = new_zb*means; 
	new_resid = zb_scale_sum* MatrixXdr::Constant(1, g.Nindv, 1); 
	MatrixXdr Xzb = new_res - new_resid; 
	Ax = vg * Xzb.transpose() / cur_snp + ve*Cx; 


	//use Ax 
        r0= b-Ax; r1 = b-Ax; 
	for(int k=0; k<colNum; k++)
	{
		r0.block(0,k,g.Nindv,1) = r0.block(0,k,g.Nindv,1).cwiseProduct(Linv); 
		r1.block(0,k,g.Nindv,1) = r1.block(0,k,g.Nindv, 1).cwiseProduct(Linv); 
	}
	
        v1 = -r0;
        while(temp>thres && iter<max_iter){
                        
		//compute Av1
		//
		MatrixXdr Cv = v1; 
		for (int k=0; k<colNum ;k ++)
			Cv.block(0, k, g.Nindv, 1) = Cv.block(0,k,g.Nindv,1).cwiseProduct( Linv); 
		MatrixXdr res(g.Nsnp,colNum);
                multiply_y_pre(Cv,colNum, res, false, exist_ind,0);
                MatrixXdr p_sum = Cv.colwise().sum();
                for(int j=0; j<g.Nsnp; j++)
			for( int k=0; k<colNum; k++)
                        res(j,k) = res(j,k)*stds(j,0);
                MatrixXdr resid(g.Nsnp, colNum);
                MatrixXdr inter = means.cwiseProduct(stds);
                resid = inter * p_sum;
                MatrixXdr zb1 = res -resid;
        	for(int k=0; k<colNum; k++)
                	zb1.block(0,k,g.Nsnp, 1)=zb1.block(0,k,g.Nsnp, 1).cwiseProduct(geno_mask);
                for(int j=0; j<g.Nsnp; j++)
			for(int k=0; k<colNum ;k++)
                        zb1(j,k) = zb1(j,k)*stds(j,0);

                MatrixXdr new_zb = zb1.transpose();
                MatrixXdr new_res(colNum, g.Nindv);
                multiply_y_post(new_zb, colNum, new_res, false, exist_ind, 0);
                MatrixXdr new_resid(colNum,g.Nindv) ;
                MatrixXdr zb_scale_sum = new_zb *means;
                new_resid = zb_scale_sum* MatrixXdr::Constant(1,g.Nindv, 1);
                MatrixXdr Xzb = new_res- new_resid;
                Av1= vg*Xzb.transpose()/cur_snp + ve*Cv;		

		for(int k=0; k<colNum; k++)
			Av1.block(0,k,g.Nindv, 1)  = Av1.block(0, k,g.Nindv, 1).cwiseProduct(Linv); 

		//
		//use Av1
		MatrixXdr temp1 = r0.transpose() * r0;
		double rr=0; 
		for(int k=0; k<colNum; k++)
			rr += temp1(k,k); 
		double vCACv=0; 
		
			
	
                MatrixXdr temp2 = v1.transpose() * Av1;
                for(int k=0; k<colNum; k++)
			vCACv += temp2(k,k); 
		t = -rr/vCACv;
		x = x + t* v1; 
		MatrixXdr r2=r1; 
		r1 =r0 - t*Av1; 
		r0=r1; 
		MatrixXdr temp3 = r1.transpose() * r1;
		double r1r1=0; 
		for(int k=0; k<colNum; k++)
			r1r1 += temp3(k,k);  
		s =  r1r1/rr; 
		v1 = - r1+ s*v1; 
                
                MatrixXdr z = r1.transpose() * r1;
                temp= z(0,0); 
		for(int k=0; k<colNum ; k++)
			if (z(k,k) > temp)
				temp = z(k,k); 
                cout<<"Iter : "<<iter <<" "<< temp <<endl;
		iter++; 
        }
}
pair<double,double> get_error_norm(MatrixXdr &c, int exist_ind, int phenoindex){
	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true,exist_ind,0);
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
                e_l(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize, phenoindex, exist_ind) - b_l(p_iter,n_iter);
                e_k(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize,phenoindex, exist_ind) - b_k(p_iter,n_iter);
            }
        }

        double ek_norm = e_k.norm();
        double el_norm = e_l.norm();
        return make_pair(ek_norm,el_norm);
    }
}



int main(int argc, char const *argv[]){

	//clock_t io_begin = clock();
    //clock_gettime (CLOCK_REALTIME, &t0);

	pair<double,double> prev_error = make_pair(0.0,0.0);
	double prevnll=0.0;

	parse_args(argc,argv);

	
	//TODO: Memory effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
	fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;
	reg = command_line_opts.reg;
	noh2g = command_line_opts.noh2g;
	//gwas=false; 
	gwas=command_line_opts.gwas; 
	cout<<"perform GWAS: "<<gwas<<endl; 
	bpheno = command_line_opts.bpheno; 
	cout<<"Binary phenotype: "<<bpheno<<endl;
	pheno_fill = command_line_opts.pheno_fill; 
	cout<<"Filling phenotype with mean: " <<pheno_fill <<endl;  
	cout<<"Compute heritability: "<<!noh2g<<endl; 
	//get number of individuals
	std::stringstream f2;
        f2 << command_line_opts.GENOTYPE_FILE_PATH << ".fam";
        g.read_fam (f2.str());
	//cout<<"ind: "<<g.Nindv<<endl; 	
	//get phenotype
	int pheno_idx = command_line_opts.pheno_idx;
	std::string filename=command_line_opts.PHENOTYPE_FILE_PATH;
        int pheno_num= read_pheno2(g.Nindv, filename, pheno_idx,pheno_fill);
	// int exist_ind =0;
       // for(int i=0; i<g.Nindv; i++)
         //       exist_ind += pheno_mask[i];
	MatrixXdr exist_ind = pheno_mask2.colwise().sum(); 
	int cov_num=0 ;
        if(filename=="")
        {
                cout<<"No Phenotype File Specified"<<endl;
                return 0 ;
        }
        cout<< "Read in "<<pheno_num << " phenotypes"<<endl;
        if(pheno_idx!=0)
                cout<<"Using phenotype "<<pheno_name[pheno_idx-1]<<endl;
	cout<< "There are "<<exist_ind<< " individuals with no missing phenotypes"<<endl; 
	MatrixXdr VarComp(pheno_num,2);

	//if(gwas)
	//	fast_mode=false; //for now, gwas need the genotype matrix, and compute kinship constructed with one chrom leave out 
	if(!reg)
		fast_mode=false; //force save whole genome if non randomized  
	if(text_version){
		if(fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH,missing);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH,missing);
	}
	else{
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH,missing,fast_mode, pheno_mask2, pheno_num);
		cout<<"read in genotype"<<endl; 	
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
	cout<<"num of random vectors: "<<B*10<<endl;  
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	float tr2= command_line_opts.tr2; 
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = true; 
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

//	geno_matrix.resize(p,n); 
//	g.generate_eigen_geno(geno_matrix, var_normalize); 

	if(!fast_mode ){
		geno_matrix.resize(p,n);
		cout<<"geno resize"<<endl; 
		g.generate_eigen_geno(geno_matrix,true,0);
		cout<<geno_matrix.rows()<<endl; 
		cout<<geno_matrix.cols()<<endl;
}
	
		
	//clock_t io_end = clock();

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
	
	ofstream c_file;
        if(debug){
                c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals_orig.txt")).c_str());
                c_file<<c<<endl;
                c_file.close();
                printf("Read Matrix\n");
        }

       #if SSE_SUPPORT==1
                if(fast_mode)
                        cout<<"Using Optimized SSE FastMultiply"<<endl;
        #endif
		//center phenotypes 
	 	MatrixXdr y_sum=pheno.colwise().sum();
		MatrixXdr exist_ind_inv(1, pheno_num); 
		for(int k=0; k<pheno_num; k++)
			exist_ind_inv(0, k) = 1/exist_ind(0,k); 
                cout<<"inverse of exist_ind: "<<exist_ind_inv<<endl; 
		MatrixXdr prevelance = y_sum.cwiseProduct(exist_ind_inv);
                MatrixXdr y_mean = y_sum.cwiseProduct(exist_ind_inv);
                for(int i=0; i<g.Nindv; i++)
		{
                        MatrixXdr temp =(pheno.block(i,0,1,pheno_num) - y_mean); //center phenotype 
                	pheno.block(i,0,1,pheno_num) = temp.cwiseProduct(pheno_mask2.block(i,0,1,pheno_num)); 
		}
		//normalize
		for(int pheno_i=0; pheno_i<pheno_num; pheno_i++){
		MatrixXdr pheno_cur = pheno.block(0,pheno_i, g.Nindv, 1); 
		MatrixXdr pheno_sum2 = pheno_cur.transpose()*pheno_cur; 
		double pheno_variance = pheno_sum2(0,0)/ (exist_ind(0,pheno_i)-1);
		pheno_variance = sqrt(pheno_variance);  
		pheno.block(0,pheno_i, g.Nindv, 1)= pheno_cur / pheno_variance; 


}
		y_sum=pheno.colwise().sum();
	
		//read in covariate 	
		std::string covfile=command_line_opts.COVARIATE_FILE_PATH;
                std::string covname=command_line_opts.COVARIATE_NAME;
                if(covfile!=""){
           	     use_cov=true;
                	cov_num=read_cov(true,g.Nindv, covfile, covname);
		 }
                else if(covfile=="")
                        cout<<"No Covariate File Specified"<<endl;


		std::string gcfile = command_line_opts.PAIR_PATH;
		int gc_pairs=0;  
		if(gcfile!=""){
			compute_gc=true; 
			gc_pairs = read_gc(gcfile);
		}
		if(pheno_num<2)
			compute_gc=false; //can not compute genetic correlation if there is only one phenotype 
	vector<double> ve_result; 
	vector<double> vg_result; 
	vector<double> vg_se; 
		 double tr_k =0 ;
                 double tr_k_rsid =0;
		if(pheno_fill && !gwas)
		{	
			//get means stds, same for all phenotypes
			for(int i=0; i<p; i++)
			{
				means(i,0)= g.get_col_mean(i, 0); 
				stds(i, 0)= 1/g.get_col_std(i, 0, exist_ind(0,0)); 
				sum2(i, 0) = g.get_col_sum2(i, 0); 
				sum(i,0) =g.get_col_sum(i, 0); 
			}
			//yKy Xy only compute once 
			yKy.resize(pheno_num,  pheno_num);
        		Xy.resize(g.Nsnp, pheno_num);
	
			if(use_cov)
                	{
		        WW = covariate.transpose() * covariate;
                        WW = WW.inverse();
	                pheno_prime= covariate.transpose()* pheno;
			}
                        compute_b1( use_cov,  y_sum, exist_ind(0,0), 0, pheno_prime, pheno_fill, pheno_num);
			//comptue tr[k]
        		tr_k = exist_ind(0,0);
			if(tr2<0 && !noh2g){
				double tr_k2=0; 
				//compute tr_k2
				rhe_reg(tr_k2, tr_k_rsid, B, 0, exist_ind(0,0)); 
				tr2= tr_k2; 
			}
		}
		if(noh2g)
		{
			
		for(int j=0; j<pheno_num; j++)
                {        for(int k=j+1; k<pheno_num; k++)
                        {
				double X = yKy(j,k) - yy(j,k); 
				double Y = yKy(j,j) - yy(j,j); 
				double Z = yKy(k,k) - yy(k,k);  
				double rg= X/sqrt(Y)/sqrt(Z); 
				cout<<"Coheritability factor estimation for phenotype: "<<j << " , " << k <<endl; 
				cout<<"rho_g: "<<rg<<endl;
				double jack_knife_se = compute_jack_knife(j,k, rg);
                        	cout<<"Jack Knife SE: "<<jack_knife_se<<endl; 
			}
		}
		return 0; 
		}
		if(pheno_fill){
		cout<<"tr_k_rsid"<<tr_k_rsid<<endl; 
		MatrixXdr A(2,2); 
		A(0,0)= tr2; 
		A(0,1) = tr_k -tr_k_rsid; 
		A(1,0) = tr_k -tr_k_rsid; 
		A(1,1) = exist_ind(0,0)-cov_num;  	
		cout<<A<<endl; 
		double vg, ve; 
		for(int i=0; i<pheno_num; i++)
		{
			cout<<"Variance Component estimation for phenotype "<<i+1<<" " <<pheno_name[i]<<" :"<<endl; 
			MatrixXdr b(2,1); 
			b(0,0) = yKy(i,i); 
			b(1,0) = yy(i,i); 
			cout<<"b: "<<endl<<b<<endl; 
			MatrixXdr herit = A.colPivHouseholderQr().solve(b);
                	cout<<"V(G): "<<herit(0,0)<<endl;
                	vg = herit(0,0);
               	 	ve = herit(1,0);
                	ve_result.push_back(ve);
			vg_result.push_back(vg); 
                	VarComp(0,0)=herit(0,0); VarComp(0,1)=herit(1,0);
                	cout<<"V(e): "<<herit(1,0)<<endl;
                	cout<<"Vp "<<herit.sum()<<endl;
                	cout<<"V(G)/Vp: "<<herit(0,0)/herit.sum()<<endl;
                	if(bpheno){
                	cout<<"Prevelance: "<<prevelance(0,0)<<endl;
                	boost::math::normal m_normal(0.0, 1.0);
                	double t = quantile(m_normal,1-prevelance(0,i));
                	double c = pdf(m_normal, t);
			c = c*c;
               	 	c= 1/c;
                	c = c* prevelance(0,i) * (1-prevelance(0,i));
                	cout<<"Liability Scale: "<<herit(0,0)*c / herit.sum()<<endl;
			}
			MatrixXdr se(1,1);
               		MatrixXdr pheno_cur = pheno.block(0,i, g.Nindv, 1);
			MatrixXdr pheno_sum2 = pheno_cur.transpose() *pheno_cur;
                	double pheno_variance = pheno_sum2(0,0) / (exist_ind(0,i)-1);
			MatrixXdr Xy_cur  = Xy.block(0,i, g.Nsnp, 1); 	
			compute_se1(Xy_cur, pheno_cur, se,vg, ve, tr2,B, exist_ind(0, i));
			vg_se.push_back(se(0,0)); 
			cout<<"phenotype variance: "<<pheno_variance<<endl;
                	cout<<"sigma_g SE: "<<se<<endl;
                	cout<<"h2g SE:"<<se/pheno_variance<<endl;
		}
		if(pheno_num>1 && pheno_fill){
                for(int j=0; j<pheno_num; j++)
                        for(int k=j+1; k<pheno_num; k++)
                        {
                        cout<<"Coheritability factor estimation for phenotype: "<<k << " , " << j <<endl;
                        MatrixXdr b(2,1);
                        b(0,0) = yKy(j, k);
                        b(1,0) = yy(j,k);
                        MatrixXdr herit = A.colPivHouseholderQr().solve(b);
                        cout <<"rho_g: "<<herit(0,0)<<endl;
                        cout <<"rho_e: "<<herit(1,0)<<endl;
			cout<<"lambda_g: "<<herit(0,0)/sqrt(vg_result[j])/sqrt(vg_result[k])<<endl; 
                        MatrixXdr se(1,1); 
			MatrixXdr pheno_cur1 = pheno.block(0,j, g.Nindv, 1); 
			MatrixXdr pheno_cur2 = pheno.block(0,k,g.Nindv, 1); 
			MatrixXdr Xy_cur1 = Xy.block(0,j,g.Nsnp, 1); 
			MatrixXdr Xy_cur2 = Xy.block(0,k,g.Nsnp, 1); 
			compute_se_coh(Xy_cur1, Xy_cur2, pheno_cur1,pheno_cur2, se,vg_result[j],vg_result[k], ve_result[j],ve_result[k], tr2,B, exist_ind(0, j),exist_ind(0,k), herit(0,0), herit(1,0), vg_se[j], vg_se[k], yKy(j,j), yy(j,j), yKy(k,k),yy(k,k));
			cout<<"SE: "<<se(0,0)<<endl;
			double rg_cur  = herit(0,0)/sqrt(vg_result[j])/sqrt(vg_result[k]); 
			double jack_knife_se = compute_jack_knife(j,k, rg_cur);  
			cout<<"Jack Knife SE: "<<jack_knife_se<<endl; 
			}
        }
	return 0;
	}
 //not filling phenotype, compute each one
 	if(!gwas){
	yKy.resize(pheno_num, pheno_num);
        Xy.resize(g.Nsnp,pheno_num);
	yy.resize(pheno_num, 1); 
	MatrixXdr tr_KA; 
	tr_KA.resize(pheno_num, pheno_num); 
	for(int it =0; it<pheno_num; it++)
		for(int pt=0; pt<pheno_num; pt++)
			tr_KA(it,pt)=0; 
	for(int pheno_i =0; pheno_i< pheno_num; pheno_i++){
		for(int i=0;i<p;i++){
                means(i,0) = g.get_col_mean(i, pheno_i);
                stds(i,0) =1/ g.get_col_std(i,pheno_i, exist_ind(0,pheno_i));
                sum2(i,0) =g.get_col_sum2(i,pheno_i);
                sum(i,0)= g.get_col_sum(i,pheno_i);
        }
		//compute current snp sum
		MatrixXdr cur_mask = pheno_mask2.block(0,pheno_i, g.Nindv, 1);
//		multiply_y_pre(cur_mask, 1, sum, false, exist_ind(0, pheno_i),pheno_i); 
		//compute current snp sum^2
//		MatrixXdr geno_mask = MatrixXdr::Zero( 1, g.Nsnp); 
//		for (int i=0; i<p; i++)
//		{
//			geno_mask(0,i)=1;
//			MatrixXdr cur_snp(1,g.Nindv); 
//			multiply_y_post(geno_mask, 1,cur_snp, false, exist_ind(0,pheno_i),pheno_i); 
//			cur_snp = cur_snp.cwiseProduct(cur_snp); 
//			cur_snp = cur_snp.cwiseProduct(cur_mask.transpose());
//			MatrixXdr tmp = cur_snp.transpose();  
//			MatrixXdr snp_sum2 = tmp.colwise().sum(); 
//			sum2(i,0)=snp_sum2(0,0);  
//			geno_mask(0,i)=0; 
//		}
//		//current pheno per snp mean
//		means = sum / exist_ind(0,pheno_i); 
//		//current pheno per snp std 
//		stds = (sum2 + exist_ind(0,pheno_i) * means.cwiseProduct(means) - 2 * sum.cwiseProduct(means))/(exist_ind(0,pheno_i)-1); 
		
//		for(int i=0; i<p; i++)
//		{
//			stds(i,0) = sqrt(stds(i,0)); 
//			stds(i,0) = 1/stds(i,0); 
//		}

		cout<<"Running on Dataset of "<<g.Nsnp<<" SNPs and "<<g.Nindv<<" Individuals"<<endl;
		cout<<"For phenotype "<< pheno_i+1<< " " <<pheno_name[pheno_i] <<" there exist "<<exist_ind(0, pheno_i) << " individuals." <<endl; 


		
	//correctness check 
	//MatrixXdr zb = MatrixXdr::Random(1, g.Nsnp); 
	//MatrixXdr res(1,g.Nindv); 
	//multiply_y_post_fast(zb, 1, res, false); 
	//cout<< MatrixXdr::Constant(1,4,1)<<endl;  
	//compute y^TKy
	

	//yKy and Xy are different for each phenotype; 
	//get updated in each iteration
	//yKy.resize(1, 1); 
	//Xy.resize(g.Nsnp,1);  
	//compute inv(WW)y  for current phenotype
	 if(use_cov)
                {
			MatrixXdr covariate_cur(g.Nindv, cov_num); 
			for(int k=0 ; k<cov_num; k++){
			MatrixXdr temp = covariate.block(0,k, g.Nindv, 1); 	
			covariate_cur.block(0,k, g.Nindv, 1) = temp.cwiseProduct(cur_mask); 
                       }
			 //compute WW
                        WW = covariate_cur.transpose() * covariate_cur;            
			WW = WW.inverse(); 
                }
	if(use_cov)
	{
		//pheno_prime.resize(cov_num, pheno_num);
		MatrixXdr pheno_cur = pheno.block(0, pheno_i, g.Nindv, 1);  
		pheno_prime= covariate.transpose()* pheno_cur; 
	}
	
	//compute yy, yKy with current pheno_mask
		if(reg) {
			MatrixXdr cur_y_sum = y_sum.block(0,pheno_i, 1,1); 
        		compute_b1( use_cov,  cur_y_sum, exist_ind(0,pheno_i), pheno_i, pheno_prime, pheno_fill, 1); 
	//		compute_b(use_cov, 1, cur_y_sum, exist_ind(0,pheno_i)); 
		}
		else
		{	
			MatrixXdr pheno_cur = pheno.block(0, pheno_i, g.Nindv, 1); 	
			yy = pheno_cur.transpose() * pheno_cur; 
			MatrixXdr temp = geno_matrix.transpose()* geno_matrix;
			yKy = pheno_cur.transpose() *temp.transpose() * pheno_cur / g.Nsnp; 
		//	cout<<"yky: "<<yKy<<endl;  
		}
	cout<<"yKy: "<<yKy<<endl; 
	cout<<"yy: "<<yy<<endl; 

	//compute tr[K]
	tr_k =0 ;
	tr_k_rsid =0; 
	tr_k = exist_ind(0,pheno_i); 

	//compute tr[K]
	MatrixXdr temp_trK = sum2 + exist_ind(0,pheno_i) * means.cwiseProduct(means) - 2* means.cwiseProduct(sum); 
	temp_trK = temp_trK.cwiseProduct(stds); 
	temp_trK = temp_trK.cwiseProduct(stds); 
	cout<<"Computed tr[K]: "<< temp_trK.sum()/g.Nsnp<<endl; 
	
	
	if(tr2<0){
	//compute/estimating tr_k2
	double tr_k2=0;
	//DiagonalMatrix<double,a> Sigma(a); 
	//Sigma.diagonal()=vec; 

	//clock_t it_begin =clock();
	if(reg){ 
	for(int i=0; i<B; i++){
		//G^T zb 
        	//clock_t random_step=clock(); 
		MatrixXdr zb= MatrixXdr::Random(g.Nindv, 10);
		zb = zb * sqrt(3); 
		for(int b=0; b<10; b++){
			MatrixXdr temp = zb.block(0,b,g.Nindv,1); 
			zb.block(0,b,g.Nindv, 1) = temp.cwiseProduct(cur_mask); 
		}
		MatrixXdr res(g.Nsnp, 10); 
		multiply_y_pre(zb,10,res, false, exist_ind(0,pheno_i),pheno_i);
		//sigma scale \Sigma G^T zb; compute zb column sum
		MatrixXdr zb_sum = zb.colwise().sum(); 
		//std::vector<double> zb_sum(10,0);  
		//res = Sigma*res;
		for(int j=0; j<g.Nsnp; j++)
		        for(int k=0; k<10;k++)
		             res(j,k) = res(j,k)*stds(j,0); 
	//	print_time();  
		//compute /Sigma^T M z_b
		MatrixXdr resid(g.Nsnp, 10);
		MatrixXdr inter = means.cwiseProduct(stds);
		resid = inter * zb_sum;
		MatrixXdr zb1(g.Nsnp,10); 
		zb1 = res - resid; // X^Tzb =zb' 
	//	clock_t Xtzb = clock(); 
		//compute zb' %*% /Sigma 
		//zb = Sigma*zb ; 
		
              	for(int k=0; k<10; k++){
                  for(int j=0; j<g.Nsnp;j++){
                        zb1(j,k) =zb1(j,k) *stds(j,0);}}
                                              
		MatrixXdr new_zb = zb1.transpose(); 
		MatrixXdr new_res(10, g.Nindv);
		multiply_y_post(new_zb, 10, new_res, false, exist_ind(0,pheno_i),pheno_i); 
		//new_res =  zb'^T \Sigma G^T 10*N 
		MatrixXdr new_resid(10, g.Nindv); 
		MatrixXdr zb_scale_sum = new_zb * means;
		new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);  
		MatrixXdr Xzb = new_res- new_resid; 
		for( int b=0; b<10; b++)
		{
			MatrixXdr temp = Xzb.block(b,0,1, g.Nindv); 
			Xzb.block(b,0,1,g.Nindv) = temp.cwiseProduct(cur_mask.transpose()); 
			for( int iter=pheno_i+1; iter<pheno_num; iter++)
			{
			MatrixXdr temp_mask = pheno_mask2.block(0,iter,g.Nindv,1); 
			MatrixXdr temp1  = temp.cwiseProduct(temp_mask.transpose()); 
			tr_KA(pheno_i, iter)+= (temp1.array() * temp1.array()).sum()/10/B/g.Nsnp/g.Nsnp; 
			}
		}
		if(use_cov)
		{
			MatrixXdr temp1 = WW * covariate.transpose() *Xzb.transpose(); 
			MatrixXdr temp = covariate * temp1;
			MatrixXdr Wzb  = zb.transpose() * temp;
			tr_k_rsid += Wzb.trace(); 
				
			Xzb = Xzb - temp.transpose(); 
		}
		tr_k2+= (Xzb.array() * Xzb.array()).sum();  
//		clock_t rest = clock(); 
	}
	tr_k2  = tr_k2 /10/g.Nsnp/g.Nsnp/B; 
	tr_k_rsid = tr_k_rsid/10/g.Nsnp/B; 
	}
	else{
	//	for(int i=0;i<n ;i++)
	//		for(int j=0; j<p; j++)
	//		{
	//			geno_matrix(j,i) = (geno_matrix(j,i)-means(j,0))*stds(j,0); 
	//		}
		cout<<"non reg"<<endl;
		MatrixXdr temp = geno_matrix.transpose()* geno_matrix; 
		temp = temp * temp ; 
		for(int i=0; i<n; i++)
			tr_k2 += temp(i,i); 
		tr_k2 = tr_k2/p/p; 
	}
		tr2=tr_k2; 
	
	}	
	MatrixXdr A(2,2); 
	A(0,0)=tr2;
	A(0,1)=tr_k-tr_k_rsid; 
	A(1,0)= tr_k-tr_k_rsid; 
	A(1,1)=exist_ind(0,0)-cov_num;
	cout<<A<<endl;   
	double vg,ve; 

	//for(int i=0; i<pheno_num; i++){
		cout<< "Variance Component estimation for phenotype "<<pheno_i+1<<" "<<pheno_name[pheno_i]<<" :"<<endl; 
		MatrixXdr b(2,1); 
		b(0,0) = yKy(pheno_i, pheno_i); 
		b(1,0) = yy(0,0);
		cout<<"b: "<<endl << b <<endl;  
		MatrixXdr herit = A.colPivHouseholderQr().solve(b); 
		cout<<"V(G): "<<herit(0,0)<<endl;
		vg = herit(0,0); 
		ve = herit(1,0);
		ve_result.push_back(ve); 
		VarComp(0,0)=herit(0,0); VarComp(0,1)=herit(1,0); 
		cout<<"V(e): "<<herit(1,0)<<endl; 
		cout<<"Vp "<<herit.sum()<<endl; 
		cout<<"V(G)/Vp: "<<herit(0,0)/herit.sum()<<endl;
		if(bpheno){
		cout<<"Prevelance: "<<prevelance(0,pheno_i)<<endl; 
 		boost::math::normal m_normal(0.0, 1.0); 
		double t = quantile(m_normal,1-prevelance(0,pheno_i)); 
		double c = pdf(m_normal, t); 
		c = c*c; 
		c= 1/c; 
		c = c* prevelance(0, pheno_i) * (1-prevelance(0, pheno_i)); 
		cout<<"Liability Scale: "<<herit(0,0)*c / herit.sum()<<endl; 
		}
	//	double c = g.Nindv* (tr2/g.Nindv - tr_k*tr_k/g.Nindv/g.Nindv); 
		
	//	cout<<"SE: "<<sqrt(2/c)<<endl; 
			

		if(reg){
		MatrixXdr se(1,1);
		MatrixXdr pheno_cur = pheno.block(0,pheno_i, g.Nindv, 1); 
	//	if(use_cov)
	//		pheno_i =pheno_prime.block(0,i,g.Nindv, 1); 
	//	MatrixXdr Xy_i = Xy.block(0, i, g.Nsnp, 1); 
		MatrixXdr pheno_sum2 = pheno_cur.transpose() *pheno_cur;
		double pheno_variance = pheno_sum2(0,0) / (exist_ind(0,pheno_i)-1); 	
	//	if(use_cov){
	//		pheno_variance  = b(1,0) /(exist_ind(0,pheno_i)-1);
	//		MatrixXdr pheno_proj = pheno_cur - covariate * WW * pheno_prime;   
	//		compute_se(Xy, pheno_proj, se, vg, ve, tr2, B, exist_ind(0, pheno_i), tr_k- tr_k_rsid, cov_num); 
	//		}
	//	else
	//		compute_se(Xy,pheno_cur,se, vg,ve,tr2,B,exist_ind(0, pheno_i),tr_k-tr_k_rsid, cov_num);

		//disable se for now
//		compute_se1(Xy, pheno_cur, se,vg, ve, tr2,B, exist_ind(0, pheno_i)); 
//		cout<<"phenotype variance: "<<pheno_variance<<endl; 
//		cout<<"sigma_g SE: "<<se<<endl; 
//		cout<<"h2g SE:"<<se/pheno_variance<<endl;
		
		}  
		if(!reg){
			MatrixXdr K = geno_matrix.transpose()* geno_matrix/p- MatrixXdr::Identity(n,n);
			MatrixXdr C= herit(0,0)* geno_matrix.transpose()*geno_matrix /p + herit(1,0)*MatrixXdr::Identity(n,n); 
			MatrixXdr temp = C*K* C *K; 
			MatrixXdr temp1 = pheno * pheno.transpose() * K * C * K; 
			double result=0; 
			double result1=0; 
			for(int i=0; i<n; i++){
				result += temp(i,i);
				result1 += temp1(i,i); 
			}
			result = result*2 +  tr2/100*herit(0,0)*herit(0,0);
			result = sqrt(result); 
			result = result / (tr2-g.Nindv);  
			result1 = result1*2 + tr2/100*herit(0,0)*herit(0,0); 
			result1 = sqrt(result1); 	
			result1 = result1 / (tr2 - g.Nindv); 
			cout<<"no random: "<<result<<endl; 
			cout<<"approximate C with yy^T: "<<result1<<endl; 
		}
	}
	
	cout<<"tr_KA: "<<endl<<tr_KA<<endl; 	
	cout<<"Computing genetic correlations... "<<endl; 
	yy= pheno.transpose() * pheno; 
	yKy = Xy.transpose() * Xy; 
	yKy = yKy/g.Nsnp; 
	cout<<"yy: "<<endl<<yy<<endl ;
	cout<<"yKy: "<<endl<<yKy<<endl; 	
	for(int pair_i=0; pair_i <pheno_num; pair_i++)
		for(int pair_j = pair_i+1; pair_j<pheno_num; pair_j++)
	{
		cout<<"Coheritability factor estimation for phenotype: "<< pair_i <<" , "<<pair_j<<endl; 
		MatrixXdr mask_i = pheno_mask2.block(0,pair_i, g.Nindv, 1);		   MatrixXdr mask_j = pheno_mask2.block(0,pair_j, g.Nindv, 1);
		MatrixXdr existN = mask_i.cwiseProduct(mask_j);  


		MatrixXdr A(2,2);
        	A(0,0)=tr_KA(pair_i, pair_j);
        	A(0,1)=existN.sum();
        	A(1,0)= existN.sum();
        	A(1,1)=existN.sum();

		MatrixXdr b(2,1); 		
		b(0,0) = yKy(pair_i, pair_j); 
		b(1,0) = yy(pair_i, pair_j); 

		MatrixXdr herit = A.colPivHouseholderQr().solve(b); 
		cout<<"rho_g: "<<herit(0,0)<<endl; 
		cout<<"rho_e: "<<herit(1,0)<<endl; 
	}

	
	}

	double vg=0.237-0.05-0.05; double ve=1-vg;//hard coded heritability 
	//vg = 0.325683; 
	if(gwas){
		for(int i=0; i<p; i++)
		{		
			means(i,0) = g.get_col_mean(i,0); 
			stds(i, 0)= 1/g.get_col_std(i, 0, exist_ind(0,0));
                        sum2(i, 0) = g.get_col_sum2(i, 0);
                        sum(i,0) =g.get_col_sum(i, 0);
		}
		//normalize phenotype
		std::string filename=command_line_opts.PHENOTYPE_FILE_PATH; 
		int pheno_sum= read_pheno2(g.Nindv, filename,pheno_idx,true); 
		MatrixXdr ysum= pheno.colwise().sum(); 
		double phenosum=ysum(0,0); 
		MatrixXdr p_i = ysum/(g.Nindv-1);
		double phenomean= p_i(0,0);  
		MatrixXdr ysum2 = pheno.transpose() * pheno; 
		double phenosum2 = ysum2(0,0); 
		double std = sqrt((phenosum2+g.Nindv*phenomean*phenomean - 2*phenosum*phenomean)/(g.Nindv-1)); 
		for(int i=0; i<g.Nindv; i++)
		{
			pheno(i,0) = pheno(i,0)-phenomean; 
			pheno(i,0) = pheno(i,0)/std; 			

		}
		cout<<"Project by covariate..."<<endl; 
		if(use_cov)
		{
			WW =  covariate.transpose() * covariate; 
			WW = WW.inverse(); 
			pheno_prime= covariate.transpose()* pheno; 
			MatrixXdr temp = WW * pheno_prime; 
			pheno = pheno - covariate * WW * pheno_prime; 
		}
		cout<<"Performing GWAS..."<<endl; 
		//perform per chromsome
		FILE *fp; 
		fp=fopen((filename+".gwas2--").c_str(), "w");
		fprintf(fp, "%s\t%s\n", "STAT", "P_VAL");
		MatrixXdr Sigmay=MatrixXdr::Zero(g.Nindv, 22); 
		MatrixXdr Chrom_start = MatrixXdr::Zero(22,2);
		int cinf_1_real=0; 
		int cinf_2_real=0;
		MatrixXdr x_guess = MatrixXdr::Zero(g.Nindv, 1);  
		MatrixXdr x_guess2 = MatrixXdr::Zero(g.Nindv,2); 
		// compute precondition for solving linear systems	
		MatrixXdr preCondition(g.Nindv, 22); 
		for(int i=0; i<g.Nindv; i++)
			for(int j=0; j<22; j++)
		{
			double temp = g.get_row_sum(i,j); 
			preCondition(i,j)= temp;	
		}
		MatrixXdr kinship_temp = preCondition.rowwise().sum();
		for(int i=0; i<g.Nindv; i++)
			for(int j=0; j<22; j++)
				preCondition(i,j) = kinship_temp(i,0)- preCondition(i,j); 	 
		////////////////////////////////////////////////////
		//compute chrom start  & end
		//Select random SNPs
		int RAN_SNP=30; 
		MatrixXdr SelectedSNP = MatrixXdr::Zero( RAN_SNP,g.Nsnp); 
		boost::mt19937 gen; 
		int select_i=0; 
		cout<<"Selecting random snps.."<<endl; 
		for(int i=0; i<22; i++)
		{
			int block_snp_num = g.get_chrom_snp(i); 
			int left=0; 
			for(int j=0; j<i; j++)
				left += g.get_chrom_snp(j); 
			int right = left + block_snp_num; 
			Chrom_start(i,0) = left; 
			Chrom_start(i,1) = right;
			//if(i==0)
			//{ 
			//	boost::random::uniform_int_distribution<> distribution(left, right); 
			//	for(int k=0; k<RAN_SNP; k++)
			//		{
			//			int idx = distribution(gen); 
			//			SelectedSNP(k, idx)=1; 
			//		}

			//}
			boost::random::uniform_int_distribution<> distribution(left, right);
                        if(i<8)
                        {
                                int idx = distribution(gen);
                                SelectedSNP(select_i,idx)=1;
                                select_i++;
                                idx = distribution(gen);
                                SelectedSNP( select_i,idx)=1;
                                select_i++;
                        }
                        else
                        {
                                int idx = distribution(gen);
                                SelectedSNP( select_i,idx)=1;
                                select_i++;
                        }
		}	
	//	cout<<SelectedSNP<<endl;
		//compute selected SNPs 
		MatrixXdr Selected(RAN_SNP, g.Nindv); 
		for(int i=0; i<RAN_SNP/10; i++){
			MatrixXdr cur_mask = SelectedSNP.block(i*10, 0, 10, g.Nsnp); 
			for(int k=0; k<10; k++)
				for(int j=0; j<g.Nsnp; j++)
				cur_mask(k,j) = cur_mask(k,j)*stds(j,0); 
			MatrixXdr x_test(10,g.Nindv);
			multiply_y_post(cur_mask, 10, x_test, false, exist_ind(0,0), 0); 
			MatrixXdr new_resid(10,g.Nindv); 
			MatrixXdr zb_scale_sum = cur_mask * means;
			new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1); 
			x_test = x_test - new_resid; 
			Selected.block(i*10, 0, 10, g.Nindv) = x_test;    	
		}
		select_i =0; 
		MatrixXdr cinf2(RAN_SNP,1);
		MatrixXdr cinf1(RAN_SNP,1);  
		for(int i=0; i<22; i++)
		{
			MatrixXdr geno_mask= MatrixXdr::Zero(g.Nsnp, 1); 
			int block_snp_num=g.get_chrom_snp(i); 
			cout<<"CHR "<<i <<": "<<block_snp_num<<endl; 
			if(block_snp_num==0)
			{
				cout<<"Chromosome "<<i+1 <<" do not have and SNP"<<endl;
				continue; 
			}
			// Matrix cur is \Sigma = \sigma_g^2 K + \sigma_e^2 I, K is made with the genotype leaving the current chromosome out
			int left=Chrom_start(i,0);
			int right = Chrom_start(i,1); 
			if(i==0)
			{
				geno_mask.block(right, 0, g.Nsnp-block_snp_num, 1) = MatrixXdr::Constant(g.Nsnp-block_snp_num, 1, 1); 
			//	cur<<geno_matrix.block(right, 0, g.Nsnp-block_snp_num,n); 
			}
			else if(i==22)
			{
				geno_mask.block(0,0, g.Nsnp-block_snp_num,1) = MatrixXdr::Constant(g.Nsnp-block_snp_num, 1,1); 
			//	cur<<geno_matrix.block(0,0, g.Nsnp-block_snp_num,n); 
			}
			else
			{
				geno_mask.block(0,0,left,1) = MatrixXdr::Constant(left, 1,1); 
				geno_mask.block(right, 0, g.Nsnp-left-block_snp_num, 1) = MatrixXdr::Constant(g.Nsnp-left-block_snp_num, 1,1); 
			//	cur<<geno_matrix.block(0,0,left,n), geno_matrix.block(right, 0, g.Nsnp-left-block_snp_num, n); 
			}
			cout<<"Left, Right"<< left <<" "<<right<<endl; 
			//MatrixXdr curInv = cur.transpose()*cur / (g.Nsnp-block_snp_num); 
			//curInv = curInv*vg; 
			//for(int j=0; j<n; j++)
			//	curInv(j,j)= curInv(j,j)+ve; 
			
			//MatrixXdr V = curInv; 
			MatrixXdr curV = preCondition.block(0,i, g.Nindv, 1);
			double chrom_vg = vg*(g.Nsnp- block_snp_num)/ g.Nsnp; 	
			double chrom_ve = 1-chrom_vg; 
			curV = curV*chrom_vg/g.Nsnp + chrom_ve* MatrixXdr::Constant(g.Nindv, 1,1); 
			curV =curV.cwiseInverse(); 
		
			//solve for y 
			MatrixXdr conj_result1(g.Nindv, 1); 
			conjugate_gradient_mailman2(g.Nindv, vg, ve, geno_mask,pheno, conj_result1, exist_ind(0,0), g.Nsnp -block_snp_num, x_guess, curV);
                        Sigmay.block(0,i,g.Nindv, 1) = conj_result1;
			x_guess = conj_result1; 
			//if(i==0){
			//selected snp are on chrom 1 
			//solve for x_test
			//for (int k=0; k<RAN_SNP/10; k++)
			//{
			//	MatrixXdr x_test = Selected.block(k*10, 0, 10, g.Nindv); 
			//	MatrixXdr x_test_t = x_test.transpose(); 
			//	MatrixXdr conj_result2(g.Nindv, 10);
                        //	conjugate_gradient_mailman2(g.Nindv,vg, ve, geno_mask, x_test_t, conj_result2, exist_ind(0,0), g.Nsnp - block_snp_num, x_guess2, curV);
			
			//	MatrixXdr cur_score1 = x_test  * conj_result1; 
			//	MatrixXdr cur_score2 = x_test * conj_result2; 


			//	for(int idx=0; idx<10; idx++)
			//	{			
			//		cinf2(k*10+idx,0) = cur_score2(idx, idx); 
			//		cinf1(k*10+idx,0) = cur_score1(idx, 0); 
			//	}
			//}
			
			//}
			if(i<8){
				MatrixXdr x_test= Selected.block(select_i, 0, 2,g.Nindv); 
				MatrixXdr x_test_t = x_test.transpose(); 
				MatrixXdr conj_result2(g.Nindv, 2);
				conjugate_gradient_mailman2(g.Nindv, vg,ve, geno_mask, x_test_t, conj_result2, exist_ind(0,0), g.Nsnp-block_snp_num, x_guess2, curV); 
				MatrixXdr cur_score1 = x_test* conj_result1; 				    MatrixXdr cur_score2 = x_test* conj_result2; 				cinf2(select_i, 0) = cur_score2(0,0); 
				cinf2(select_i+1, 0) = cur_score2(1,1); 
				cinf1(select_i, 0) = cur_score1(0,0); 
				cinf1(select_i+1, 0) = cur_score1(1,0); 
				select_i = select_i+ 2; 
			} 
			else{
				if(i==8)
					x_guess2 = x_guess2.block(0,0,g.Nindv, 1); 
				MatrixXdr x_test = Selected.block(select_i, 0, 1, g.Nindv); 
				MatrixXdr x_test_t = x_test.transpose(); 
				MatrixXdr conj_result2(g.Nindv, 2) ; 
				conjugate_gradient_mailman2(g.Nindv, vg, ve, geno_mask, x_test_t, conj_result2, exist_ind(0,0), g.Nsnp-block_snp_num, x_guess2, curV);
				MatrixXdr cur_score1 = x_test* conj_result1; 
				MatrixXdr cur_score2 =x_test* conj_result2; 
				cinf2(select_i, 0 ) = cur_score2(0,0); 
				cinf1(select_i, 0) = cur_score1(0,0); 
				select_i++;  
			}
		}
		double cinf_real = (cinf_1_real/g.Nsnp) / (cinf_2_real/g.Nsnp); 
		cout<<"Real cinf: "<< cinf_real<<endl;
		//MatrixXdr XSigmay = geno_matrix * Sigmay;
		MatrixXdr XSigmay = Selected * Sigmay; // 30*g.Nindv, X   g.Nindv *22 
		double cinf_1 = 0;
		double cinf_2 = 0; 
		double cinf_3=0;  


		select_i=0; 
		for(int i=0; i<RAN_SNP; i++)
		{
				
				cinf_1 += cinf1(i, 0)*cinf1(i, 0); 
				cinf_2 += cinf1(i, 0)*cinf1(i,0) / cinf2(i,0); 
				cinf_3 += cinf2(i,0); 
		}
		cinf_1 = cinf1.cwiseProduct(cinf1).sum()/RAN_SNP; 
		MatrixXdr temp=cinf2.cwiseInverse(); 
		temp = temp.cwiseProduct(cinf1); 
		temp = temp.cwiseProduct(cinf1); 
		cinf_2 = temp.sum()/RAN_SNP; 
		double cinf = cinf_1 / cinf_2;
		cout<<"Mean of Xtext^T V-1 Xtest: "<<cinf_3/RAN_SNP<<endl;
                cout<< "Mean of X-2_score: "<<cinf_2<<endl;
		cout<<"Cinf: "<<cinf<<endl;  
		int snp_idx=0; 
		//Compute X^T V-1 y 
		MatrixXdr XVy(g.Nsnp, 23);
                for( int k=0; k<22; k++)
                {
                        MatrixXdr zb = Sigmay.block(0, k, g.Nindv,1);
                        MatrixXdr res(g.Nsnp, 1);
                        multiply_y_pre(zb, 1, res, false, exist_ind(0,0), 0);
                        MatrixXdr zb_sum = zb.colwise().sum();
                        for(int j=0; j<g.Nsnp; j++)
                                res(j,0) = res(j,0)*stds(j,0);
                        MatrixXdr resid(g.Nsnp, 1);
                        MatrixXdr inter = means.cwiseProduct(stds);
                        resid = inter * zb_sum;
                        MatrixXdr zb1 = res- resid;
                        XVy.block(0,k,g.Nsnp, 1) = zb1;
                }




		for(int i=0; i<22; i++)
                {
                        int block_snp_num =g.get_chrom_snp(i);
                        for(int j=0; j<block_snp_num; j++)
                        {
                                double test_score = XVy(snp_idx, i)* XVy(snp_idx,i)/ cinf;
//                                cout<<"snp "<<snp_idx<<" : "<<test_score << " ";
                                boost::math::chi_squared mydist(1);
                                double p = boost::math::cdf(mydist, test_score);
  //                              cout<<"stat: "<<1-p <<endl;
                                fprintf(fp, "%g\t%g\n",test_score, 1-p);
                                snp_idx++;

                        }
                }





		fclose(fp); 
 
 
	} 
	//clock_t it_end = clock();
	
		
	//clock_t total_end = clock();
//	double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
//	double avg_it_time = double(it_end - it_begin) / (B * 1.0 * CLOCKS_PER_SEC);
//	double total_time = double(total_end - total_begin) / CLOCKS_PER_SEC;
//	cout<<"IO Time:  "<< io_time << "\nAVG Iteration Time:  "<<avg_it_time<<"\nTotal runtime:   "<<total_time<<endl;

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
