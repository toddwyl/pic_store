#! coding:utf-8
import urllib.parse
import argparse
import pysnooper
# zhihu_gs_prefix = r'https://www.zhihu.com/equation?tex='
# latex_gs = r'\left\{\left(\bar{\phi}_{n}, \bar{M}_{n}\right)=g\left(a_{n}\right)\right\}_{n=1}^{N}'
# zhihu_gs = zhihu_gs_prefix + urllib.parse.quote(latex_gs)

def convert_a_math(latex_gs):
    zhihu_gs_prefix = r'https://www.zhihu.com/equation?tex='
    zhihu_gs = zhihu_gs_prefix + urllib.parse.quote_plus(latex_gs)
    return zhihu_gs

def read_md(file_path):
    md_file = open(file_path, 'r') 
    md_str = md_file.read()
    return list(md_str)

@pysnooper.snoop('C:\\Users\\47181\\Documents\\log')
def conver_str(md_str):
    stack_dollar=[]
    flag_one_begin=False # first $
    flag_two_begin=False # second $$
    flag_continue=False
    # print('md_str:\n',''.join(md_str))
    md_str_new = []
    for i,s in enumerate(md_str):            
        print(''.join(md_str_new))
        look_ahead='?'
        if i!=len(md_str)-1:
            look_ahead = md_str[i+1]
        if flag_continue:
            flag_continue = False
            continue
        if not flag_one_begin: # not in the match state
            if s!='$':
                md_str_new.append(s)
            elif s=='$':
                if look_ahead=='$':
                    flag_one_begin = True
                    flag_two_begin = True
                    flag_continue = True
                    continue
                elif look_ahead!='$':
                    flag_one_begin = True
                    flag_two_begin = False
                    continue      
        elif flag_one_begin: # case for $ math $ (in the match state)
            if not flag_two_begin:
                if s == '$':
                    flag_one_begin = False
                    # get a $ math $ match, now we convert stack_dollar:
                    latex_zhihu = convert_a_math(''.join(stack_dollar))
                    insert_len = len(latex_zhihu)
                    md_str_new.extend(list(latex_zhihu))
                    stack_dollar=[]
                else:
                    # if s!=' ':
                    stack_dollar.append(s)
            elif flag_two_begin: # case for $$ math $$ (in the match state)
                if s == '$':
                    flag_one_begin = False
                    flag_two_begin = False
                    flag_continue = True # miss the second ending dollar
                    # get a $ math $ match, now we convert stack_dollar:
                    latex_zhihu = convert_a_math(''.join(stack_dollar))
                    insert_len = len(latex_zhihu)
                    md_str_new.extend(list(latex_zhihu))
                    stack_dollar=[]
                else:
                    # if s!=' ':
                    stack_dollar.append(s)
    return ''.join(md_str_new)
            
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', help='The md file will input to convert for zhihu_type',default=r'K:\blog\pic_store\README.md')
    args = parser.parse_args()

    md_str = read_md(args.fpath)
    print(conver_str(md_str))
    # print(convert_a_math(r' X_y '))