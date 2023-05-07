
from cgi import print_form
import re
from tkinter.messagebox import RETRY
from sklearn import metrics
from hmeasure import h_score 
import numpy as np
import os
import pandas as pd

class Evaluater(object):
    def __init__(self,log_name):
        super(Evaluater).__init__()
        self.log_name=log_name
        self.log_path=os.path.join("log",log_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.f=open(os.path.join(self.log_path,"log.txt"),"w")
        self.count=0
        self.auc_sum=0
        self.ap_sum=0
        self.acc_sum=0
        self.rec_sum=0
        self.prec_sum=0
        self.f1_sum=0
        self.h_sum=0
        self.g_sum=0
        self.type1_sum=0
        self.type2_sum=0
    
    def clc_update(self,label,pred_prob,pred_res):
        self.count+=1
        precision,recall,th=metrics.precision_recall_curve(label,pred_prob,pos_label=1)
        pr_csv={"{}_p".format(self.log_name):precision,"{}_r".format(self.log_name):recall}
        pr_csv=pd.DataFrame(pr_csv)
        pr_csv.to_csv(os.path.join(self.log_path,"{}_pr_{}.csv".format(self.log_name,self.count)))
        fpr, tpr, thresholds = metrics.roc_curve(label, pred_prob, pos_label=1)
        ftpr_csv={"{}_fpr".format(self.log_name):fpr,"{}_tpr".format(self.log_name):fpr}
        ftpr_csv=pd.DataFrame(ftpr_csv)
        ftpr_csv.to_csv(os.path.join(self.log_path,"{}_ftpr_{}.csv".format(self.log_name,self.count)))

        auc=metrics.auc(fpr,tpr)
        ap=metrics.average_precision_score(label,pred_prob)
        
        ########需要01值计算的指标 acc,prec,rec,f1,type1 error type2 error##########
        eval_range_th={"acc":[],"rec":[],"prec":[],"f1":[],"type1":[],"type2":[],"h_measure":[],"g_score":[]}

        for score in np.linspace(0.05,0.95,19):
            pred_res_tmp=(pred_prob>score)*1
            acc_tmp =metrics.accuracy_score(label,pred_res_tmp)
            rec_tmp =metrics.recall_score(label,pred_res_tmp)
            prec_tmp=metrics.precision_score(label,pred_res_tmp)
            f1_tmp  =metrics.f1_score(label,pred_res_tmp)
            tn, fp, fn, tp = metrics.confusion_matrix(label, pred_res_tmp).ravel()
            type_1_tmp=fp/(fp+tn)
            type_2_tmp=fn/(tp+fn)
            h_measure_tmp=h_score(label, pred_res_tmp)
            g_score_tmp=np.sqrt(tp*tn/((tp+fn)*(tn+fp)))

            eval_range_th["acc"].append(acc_tmp)
            eval_range_th["rec"].append(rec_tmp)
            eval_range_th["prec"].append(prec_tmp)
            eval_range_th["f1"].append(f1_tmp)
            eval_range_th["type1"].append(type_1_tmp)
            eval_range_th["type2"].append(type_2_tmp)
            eval_range_th["h_measure"].append(h_measure_tmp)
            eval_range_th["g_score"].append(g_score_tmp)
        eval_csv=pd.DataFrame(eval_range_th)
        eval_csv.to_csv(os.path.join(self.log_path,"{}_eval_all_{}.csv".format(self.log_name,self.count)))


        acc =metrics.accuracy_score(label,pred_res)
        rec =metrics.recall_score(label,pred_res)
        prec=metrics.precision_score(label,pred_res)
        f1  =metrics.f1_score(label,pred_res)
        tn, fp, fn, tp = metrics.confusion_matrix(label, pred_res).ravel()
        type_1=fp/(fp+tn)
        type_2=fn/(tp+fn)

        h_measure=h_score(label, pred_res)
        g_score=np.sqrt(tp*tn/((tp+fn)*(tn+fp)))

        self.auc_sum+=auc
        self.ap_sum+=ap
        self.acc_sum+=acc
        self.rec_sum+=rec
        self.prec_sum+=prec
        self.f1_sum+=f1
        self.type1_sum+=type_1
        self.type2_sum+=type_2
        self.h_sum+=h_measure
        self.g_sum+=g_score
        print("count:%d|auc:%.3f,ap:%.3f,acc:%.3f,prec:%.3f,rec%.3f,f1:%.3f,type1:%.3f,type2:%.3f,h_measure:%.3f,g_mean:%.3f"%(self.count,auc,ap,acc,prec,rec,f1,type_1,type_2,h_measure,g_score))
        self.f.write("count:%d|auc:%.3f,ap:%.3f,acc:%.3f,prec:%.3f,rec:%.3f,f1:%.3f,type1:%.3f,type2:%.3f,h_measure:%.3f,g_mean:%.3f\n"%(self.count,auc,ap,acc,prec,rec,f1,type_1,type_2,h_measure,g_score))
        return {"aun":auc,"ap":ap,"acc":acc,"prec":prec,"rec":rec,"f1":f1,"type1_e":type_1,"type2_e":type_2,"h_measure":h_measure,"g_mean":g_score}

    def mean(self):
        auc_mean=self.auc_sum/self.count
        ap_mean=self.ap_sum/self.count
        acc_mean=self.acc_sum/self.count
        prec_mean=self.prec_sum/self.count
        rec_mean=self.rec_sum/self.count
        f1_mean=self.f1_sum/self.count
        type1_mean=self.type1_sum/self.count
        type2_mean=self.type2_sum/self.count
        h_mean=self.h_sum/self.count
        g_mean=self.g_sum/self.count
        self.f.write("mean|auc:%.3f,ap:%.3f,acc:%.3f,prec:%.3f,rec%.3f,f1:%.3f,type1:%.3f,type2:%.3f,h_measure:%.3f,g_mean:%.3f\n"%(auc_mean,ap_mean,acc_mean,prec_mean,rec_mean,f1_mean,type1_mean,type2_mean,h_mean,g_mean))
        print("mean|auc:%.3f,ap:%.3f,acc:%.3f,prec:%.3f,rec%.3f,f1:%.3f,type1:%.3f,type2:%.3f,h_measure:%.3f,g_mean:%.3f\n"%(auc_mean,ap_mean,acc_mean,prec_mean,rec_mean,f1_mean,type1_mean,type2_mean,h_mean,g_mean))
        
        res={"auc_mean":self.auc_sum/self.count,
             "ap_mean": self.ap_sum/self.count,
             "acc_mean":self.acc_sum/self.count,
             "prec_mean":self.prec_sum/self.count,
             "rec_mean":self.rec_sum/self.count,
             "f1_mean":self.f1_sum/self.count,
             "type1_mean":self.type1_sum/self.count,
             "type2_mean":self.type2_sum/self.count,
             "h_mean":self.h_sum/self.count,
             "g_mean":self.g_sum/self.count}
        return res

    def __def__(self):
        self.f.close()

    def clean(self):
        self.count=0
        self.auc_sum=0
        self.ap_sum=0
        self.acc_sum=0
        self.rec_sum=0
        self.prec_sum=0
        self.f1_sum=0
        self.h_sum=0
        self.g_sum=0
        self.type1_sum=0
        self.type2_sum=0
