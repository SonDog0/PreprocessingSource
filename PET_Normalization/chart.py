import pandas as pd
import re
import logging
from datetime import datetime


def sortingList(lst):
    lst = [str(s) for s in lst]
    rtnList = sorted(lst, key=len , reverse=True)
    return rtnList


def StringToList_df(list1, list2, list3):
    result = []
    for a in list1:
        for b in list2:
            result.append([a, b, list3])

    return pd.DataFrame(result)

if __name__ == '__main__':
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    mylogger.addHandler(stream_hander)

    file_handler = logging.FileHandler('0805TestSet_chart.log')
    mylogger.addHandler(file_handler)

    mylogger.info("start chart normalization at {} ".format(datetime.now()))



    try :


        # linux
        diagnosis_path = '/Apps/pet_pipeline/testDataSet/test_diag.csv'
        assessment_path = '/Apps/pet_pipeline/testDataSet/test_ass.csv'
        info_path = '/Apps/pet_pipeline/testResult/testR_info.csv'
        dict_chart = '/Apps/pet_pipeline/dict/official/Chart_표준화_200706.xlsx'
        kisti_exploding_path = '/Apps/pet_pipeline/dict/pl_chart/kisti_exploding.csv'
        disease_exploding_path = '/Apps/pet_pipeline/dict/pl_chart/disease_exploding.csv'
        symptom_exploding_path = '/Apps/pet_pipeline/dict/pl_chart/symptom_exploding.csv'

        result = '/Apps/pet_pipeline/testResult/testR_chart.csv'




        # Phase 1

        # diagnosis 테이블 load
        diagnosis = pd.read_csv(diagnosis_path , encoding='CP949')
        diagnosis.drop('chart_memo', axis=1 , inplace=True)
        print(len(diagnosis))
        # assessment 테이블 load
        assessemnt = pd.read_csv(assessment_path , encoding='CP949')

        # assessment 값 있으면 dg값을 대체
        assessemnt = assessemnt.groupby('num_diag')['dx'].apply(lambda x: ','.join(x.astype(str))).reset_index()

        diagnosis.loc[diagnosis.num.isin(assessemnt.num_diag), ['chart_cc']] = assessemnt.loc[assessemnt.num_diag.isin(diagnosis.num), ['dx']].values

        # exploding
        diagnosis['chart_id'] = diagnosis['chart_id'].fillna("-1")
        diagnosis['chart_cc'] = diagnosis['chart_cc'].fillna("none")


        diagnosis.chart_id = diagnosis.chart_id.astype('int64')
        newStr = diagnosis.chart_cc.str.split(',|/').tolist()

        newDiagnosis = pd.DataFrame(newStr, index=[diagnosis.num ,diagnosis.num_info, diagnosis.hospital_id, diagnosis.chart_id , diagnosis.chart_modified]).stack()
        newDiagnosis = newDiagnosis.reset_index([0, 'num', 'num_info', 'hospital_id','chart_id' , 'chart_modified'])
        newDiagnosis.columns = ['num', 'num_info', 'hospital_id','chart_id' , 'chart_modified' , 'chart_cc']
        newDiagnosis['chart_cc'] = newDiagnosis['chart_cc'].str.replace(" ","")
        newDiagnosis['chart_cc'] = newDiagnosis['chart_cc'].str.replace(".","")

        print(len(newDiagnosis))
        # # newDiagnosis.to_csv("[20200324]diagnosis_assessemnt_exploding.csv" , encoding = 'CP949', index = False)




        # Phase 2

        # dict 만들기 ( KISTI + CHART_EXCEL )

        # 1.  KISTI exploding

        df_kisti= pd.read_excel(dict_chart , sheet_name= 3)
        df_kisti = df_kisti[['표준용어변환','표준용어변환 (한글)']]
        df_kisti.rename(columns = {'표준용어변환' : 'eng'} , inplace  = True)
        df_kisti.rename(columns = {'표준용어변환 (한글)' : 'kor'}, inplace  = True)
        df_kisti.dropna(how='all', inplace = True)
        df_kisti['kor'] = df_kisti['kor'].str.replace(" ","")
        df_kisti['eng'] = df_kisti['eng'].str.replace(" ","")

        # df_kisti # eng , kor 원본데이터
        ### dict_kisti -> kisti exploding
        dict_kisti = pd.DataFrame()

        for el in range(len(df_kisti)):
            kisti_new = df_kisti.loc[el, 'kor']
            kor_el = df_kisti.loc[el, 'kor'].split(',')
            eng_el = df_kisti.loc[el, 'eng'].split(',')
            dict_kisti = pd.concat([dict_kisti, StringToList_df(kor_el,eng_el , kisti_new)], axis=0).reset_index(drop=True)


        dict_kisti.columns = ['kor', 'eng','std']
        dict_kisti.to_csv(kisti_exploding_path, encoding='CP949' , index = False)
        kor_kisti_list = list(dict_kisti.loc[:, 'kor'])
        eng_kisti_list = list(dict_kisti.loc[:, 'eng'])


        dict_chartExcel_Symptom = pd.read_excel(dict_chart,sheet_name= 1)
        dict_chartExcel_Symptom = dict_chartExcel_Symptom[['대표증상명','증상목록']]
        dict_chartExcel_Symptom = dict_chartExcel_Symptom.dropna(how='all')
        dict_chartExcel_Symptom = dict_chartExcel_Symptom.fillna(method='ffill')
        dict_chartExcel_Symptom.rename(columns = {'대표증상명' : 'main'} , inplace  = True)
        dict_chartExcel_Symptom.rename(columns = {'증상목록' : 'sub'}, inplace  = True)
        dict_chartExcel_Symptom['main'] = dict_chartExcel_Symptom['main'].str.replace(" ","")
        dict_chartExcel_Symptom['sub'] = dict_chartExcel_Symptom['sub'].str.replace(" ","")
        dict_chartExcel_Symptom.to_csv(symptom_exploding_path, encoding = 'CP949', index= False)


        dict_chartExcel_Disease = pd.read_excel(dict_chart,sheet_name= 0)
        dict_chartExcel_Disease = dict_chartExcel_Disease[['대표질병명','질병목록']]
        dict_chartExcel_Disease = dict_chartExcel_Disease.dropna(how='all')
        dict_chartExcel_Disease = dict_chartExcel_Disease.fillna(method='ffill')
        dict_chartExcel_Disease.rename(columns = {'대표질병명' : 'main'} , inplace  = True)
        dict_chartExcel_Disease.rename(columns = {'질병목록' : 'sub'}, inplace  = True)
        dict_chartExcel_Disease['main'] = dict_chartExcel_Disease['main'].str.replace(" ","")
        dict_chartExcel_Disease['sub'] = dict_chartExcel_Disease['sub'].str.replace(" ","")
        dict_chartExcel_Disease.to_csv(disease_exploding_path, encoding = 'CP949', index= False)


        main_chartD_list = list(dict_chartExcel_Disease.loc[:, 'main'])
        sub_chartD_list = list(dict_chartExcel_Disease.loc[:,'sub'])

        main_chartS_list = list(dict_chartExcel_Symptom.loc[:, 'main'])
        sub_chartS_list = list(dict_chartExcel_Symptom.loc[:,'sub'])

        dict_list = ['kor_kisti_list' , 'eng_kisti_list' , 'main_chartD_list' , 'sub_chartD_list' , 'main_chartS_list' , 'sub_chartS_list']
        dict_list_content = []


        for i in range (0, len(dict_list)) :
            dict_list_content.append(sortingList(eval(dict_list[i])))

        # 2. 각 컬럼을 리스트로 만들어서 concat - > 이후에 변환
        tot_dict_list = []
        for i in range (0, len(dict_list_content)):
            tot_dict_list.extend(eval(str(dict_list_content[i])))

        tot_dict_list = [re.sub(' ','' ,s ) for s in tot_dict_list]
        tot_dict_list = [str(x).lower() for x in tot_dict_list]



        # Phase 3

        # 표준화 ( dict contain in CHARTDB( diagnosis.chartcc ) )

        raw_dx = newDiagnosis.loc[:,'chart_cc']
        raw_dx = [str(x).lower() for x in raw_dx]
        raw_dx = [str(x).strip() for x in raw_dx]
        raw_dx = [re.sub(r'\([^)]*\)', '', s) for s in raw_dx]
        raw_dx = [re.sub(' ','' ,s ) for s in raw_dx]
        raw_dx = [re.sub('([ㄱ-ㅎㅏ-ㅣ]+)','',s) for s in raw_dx]

        len_raw_dx = len(raw_dx)
        # print('total LENGTH : ' , len(raw_dx))

        cnt = 0
        nor_dx = []

        import jellyfish

        for i in range (0 , len(raw_dx)):
            dx = ''
            if i % 10000== 0:
                print('process : ', i)
                print('process percentage : ', round((i / len_raw_dx) * 100, 2))
            for j in range (0 , len(tot_dict_list)):
                calcSim = jellyfish.jaro_distance(tot_dict_list[j], raw_dx[i])
                if len(str(tot_dict_list[j])) > 2 and str(tot_dict_list[j]) in str(raw_dx[i]):
                    dx =tot_dict_list[j]
                    cnt += 1
                    break

                elif len(str(tot_dict_list[j])) <= 2 and str(tot_dict_list[j]) == str(raw_dx[i]):

                    dx =tot_dict_list[j]
                    cnt += 1
                    break


                elif calcSim > 0.9 :
                    dx = tot_dict_list[j]
                    cnt +=1
                    break

                else:
                    dx = 'unknown'

            nor_dx.append(dx)

        print('1 : 1 바뀐 항목 수 : ' , cnt)

        print('Percent : ' , round( cnt  / len(raw_dx) * 100 , 2))

        print('total LENGTH : ' , len(raw_dx))

        # newDiagnosis.to_csv('newDiagnosisGT3andJF_edit.csv' , encoding='CP949' , index= False)


        # Phase 4

        # DICT MAPPING
        # D/S expend 시키고 , nor_dx == 'sub' 면 , main으로 변경

        main_chart_list = []
        main_chart_list.extend(main_chartS_list)
        main_chart_list.extend(main_chartD_list)


        sub_chart_list = []
        sub_chart_list.extend(sub_chartS_list)
        sub_chart_list.extend(sub_chartD_list)


        sub_chart_list = [re.sub(' ','' ,s ) for s in sub_chart_list]
        sub_chart_list = [str(x).lower() for x in sub_chart_list]


        regex_eng_kisti_list = [re.sub(' ','' ,s ) for s in eng_kisti_list]
        regex_eng_kisti_list = [str(x).lower() for x in regex_eng_kisti_list]



        nor_dx_map = []
        mapping = ''

        map_cnt = 0
        for i in range(len(nor_dx)):
            map_cnt +=1
        #     print(map_cnt)
            for j in range(len(sub_chart_list)):
                if nor_dx[i] == sub_chart_list[j]:
                    mapping = main_chart_list[j]
                    break
                else:
                    mapping = nor_dx[i]
            nor_dx_map.append(mapping)

        nor_dx_map_kor = []
        lang_mappng = ''

        for i in range ( len(nor_dx_map)):
            for j in range(len(eng_kisti_list)):
                if nor_dx[i] == regex_eng_kisti_list[j]:
                    lang_mappng = kor_kisti_list[j]
                    break
                else :
                    lang_mappng = nor_dx_map[i]
            nor_dx_map_kor.append(lang_mappng)



        newDiagnosis['nor_dx_map_kor'] = nor_dx_map_kor


        # newDiagnosis.to_csv('/home/son/army/pet/data/chart_cc_data/newDiagnosisGT3andJF_edit_map_regex_kor_0723.csv' , encoding='CP949' , index= False)


        # Phase 5
        # isDisease
        # newDiagnosis = pd.read_csv('/home/son/army/pet/data/chart_cc_data/newDiagnosisGT3andJF_edit_map_regex_kor_0723.csv' , encoding='CP949')


        dict_chartExcel_Disease= pd.read_csv(disease_exploding_path , encoding='CP949')
        dict_chartExcel_Symptom= pd.read_csv(symptom_exploding_path , encoding='CP949')
        dict_kisti= pd.read_csv(kisti_exploding_path, encoding='CP949').dropna()

        main_D = dict_chartExcel_Disease.loc[:,'main']
        sub_D = dict_chartExcel_Disease.loc[:,'sub']

        main_S = dict_chartExcel_Symptom.loc[:,'main']
        sub_S = dict_chartExcel_Symptom.loc[:,'sub']

        kisti = dict_kisti.loc[:,'kor']
        kisti = [re.sub(' ','' , x)  for x in kisti]

        Disease = []
        Symptom = []

        Disease.extend(main_D)
        Disease.extend(sub_D)

        Symptom.extend(main_S)
        Symptom.extend(sub_S)

        Disease = [re.sub(' ','' , x)  for x in Disease]
        Symptom = [re.sub(' ','' , x)  for x in Symptom]


        chartcc = newDiagnosis.loc[:,'nor_dx_map_kor']

        chartcc = [re.sub(' ','' , x)  for x in chartcc]

        isDisease = []

        for i in range ( len (chartcc) ):
            if chartcc[i] in Disease or chartcc[i] in kisti :
                isDisease.append(0)
            elif chartcc[i] in Symptom:
                isDisease.append(1)

            else:
                isDisease.append(-1)

        newDiagnosis['isDisease'] = isDisease

        newDiagnosis.rename(columns = {'num_info':'info_origin_num' ,'num' :'chart_origin_num', 'nor_dx_map_kor' : 'dx' , 'chart_modified' : 'charted' } , inplace=True)

        newDiagnosis.insert(0, 'num' , range(0, len(newDiagnosis)))



        newDiagnosis = newDiagnosis[['num', 'chart_origin_num', 'info_origin_num', 'isDisease', 'dx' , 'charted' ]]

        info_df = pd.read_csv(info_path , encoding='CP949' )

        info_df = info_df[['info_origin_num' , 'birth' , 'species_code']]

        newDiagnosis.to_csv('0803chart_test.csv', encoding='CP949' , index=False)

        new_df = pd.merge(left=newDiagnosis , right=info_df , left_on='info_origin_num' , right_on='info_origin_num' , how= 'inner')


        def calc_lifecycle(data):
            # 빈 리스트 생성
            pet_age = []
            life_cycle = []

            # data 에서 리스트 추출
            chart_modified_list = data['charted']
            pet_birth_list = data['birth']
            species_code_list = data['species_code']

            # 예외 항목 설정
            exception = [None, 'unknown', '']

            for i in range(len(data)):
                # 예외 항목이면 unknown으로 처리
                if (chart_modified_list[i] in exception) or (pet_birth_list[i] in exception) or (
                        pd.isna(chart_modified_list[i]) == True) or (pd.isna(pet_birth_list[i]) == True):
                    pet_age.append('unknown')
                    life_cycle.append('unknown')

                else:
                    # 진료 당시 나이 계산 (Month)
                    end_date_y = int(chart_modified_list[i][:4])
                    end_date_m = int(chart_modified_list[i][5:7])
                    end_date_d = int(chart_modified_list[i][8:10])

                    start_date_y = int(pet_birth_list[i][:4])

                    if (start_date_y == 1899):
                        pet_age.append('unknown')
                        life_cycle.append('unknown')
                        continue

                    start_date_m = int(pet_birth_list[i][5:7])
                    start_date_d = int(pet_birth_list[i][8:10])

                    num_months = (end_date_y - start_date_y) * 12 + (end_date_m - start_date_m)

                    if (num_months < 0) or (num_months >= 300):
                        pet_age.append('unknown')
                        life_cycle.append('unknown')
                        continue

                    if (end_date_d - start_date_d) > 0:
                        num_months += 1
                    pet_age.append(num_months)

                    # 개 or 고양이일 때 life cycle 추가
                    if species_code_list[i] == 1 :
                        if num_months <= 6:
                            life_cycle.append('유년기')
                        elif num_months <= 24:
                            life_cycle.append('청년기')
                        elif num_months <= 84:
                            life_cycle.append('장년기')
                        else:
                            life_cycle.append('노년기')

                    elif species_code_list[i] == 2 :
                        if num_months < 7:
                            life_cycle.append('유년기')
                        elif num_months <= 24:
                            life_cycle.append('청년기')
                        elif num_months <= 72:
                            life_cycle.append('청장년기')
                        elif num_months <= 120:
                            life_cycle.append('장년기')
                        elif num_months <= 144:
                            life_cycle.append('노년기')
                        else:
                            life_cycle.append('노년기후반')

                    elif species_code_list[i] == -1:
                        life_cycle.append('unknown')


            data['pet_age'] = pet_age
            data['life_cycle'] = life_cycle

            return data

        new_df_fin = calc_lifecycle(new_df)


        new_df_fin.rename(columns = { 'life_cycle':'lifecycle' } , inplace=True)

        new_df_fin = new_df_fin[['num' , 'chart_origin_num', 'info_origin_num' ,'isDisease' , 'lifecycle' , 'charted', 'dx']]


        # kisti 추가 매핑

        new_nor_dx = list(new_df_fin.loc[:,'dx'])
        chart_cc = list(newDiagnosis.iloc[:,4])
        dict_kisti_kor = list(dict_kisti.loc[:,'kor'])
        dict_kisti_eng = list(dict_kisti.loc[:,'eng'])
        dict_kisti_map = list(dict_kisti.loc[:,'std'])

        nor_dx0512 = []
        newstr0512 = ''

        for i in range ( 0 , len(new_nor_dx)):
            for j in range ( 0 , len(dict_kisti_eng)):
                if new_nor_dx[i] == dict_kisti_eng[j] or new_nor_dx[i] == dict_kisti_kor[j]:
                    newstr0512 = dict_kisti_map[j]
                    break
                else:
                    newstr0512 = new_nor_dx[i]
            nor_dx0512.append(newstr0512)

        print(nor_dx0512)
        new_df_fin['dx_concat'] = nor_dx0512

        ###
        new_df_fin.drop(columns = ['dx'], inplace= True)
        new_df_fin.rename(columns = {'dx_concat' : 'dx'}, inplace = True)



        # start 16:24

        # 1 : 1 바뀐 항목 수 :  248945
        # Percent :  23.49
        # total LENGTH :  1059593

        # 1 : 1 바뀐 항목 수 :  198320
        # Percent :  18.72
        # total LENGTH :  1059593


        # jf
        # 1 : 1 바뀐 항목 수 :  205942
        # Percent :  19.44
        # total LENGTH :  1059593

        # 대소문자
        # 1 : 1 바뀐 항목 수 :  220816
        # Percent :  20.84
        # total LENGTH :  1059593

        # 자모음제거
        # 1 : 1 바뀐 항목 수 :  220872
        # Percent :  20.84
        # total LENGTH :  1059593

        # NEW DICT
        # process percentage :  100.0
        # 1 : 1 바뀐 항목 수 :  220001
        # Percent :  20.76
        # total LENGTH :  1059593


        # Phase 6
        # 대분류 추가
        df_kisti= pd.read_excel(dict_chart,sheet_name= 3)


        df_kisti_with_Main = df_kisti[['대분류','표준용어변환 (한글)']]
        pet_input_chart = new_df_fin

        # list 추출
        dx_list = pet_input_chart.loc[:,'dx'].tolist()
        data_han = df_kisti_with_Main.loc[:,'표준용어변환 (한글)'].tolist()
        data_std = df_kisti_with_Main.loc[:,'대분류'].tolist()

        import re
        dx_list = [re.sub(' ','' , x)  for x in dx_list]
        data_han = [re.sub(' ','' , str(x))  for x in data_han]
        data_std = [re.sub(' ','' , str(x))  for x in data_std]

        new_str = ''
        new_arr = []


        for i in range(len(dx_list)):
            if i%10000 == 0:
                    print('process : ', i)
                    print('process percentage : ', round((i / len(dx_list)) * 100, 2))
            for j in range(len(data_han)):
                if dx_list[i] == 'unknown':
                    new_str = 'unknown'
                    break

                if dx_list[i] == data_han[j]:
                    new_str = data_std[j]
                    break
                else:
                    new_str = 'check'

            new_arr.append(new_str)


        pet_input_chart['main_category'] = new_arr
        pet_input_chart.to_csv(result, encoding = 'CP949')


    except BaseException as e :
        mylogger.info(str(e))

    finally:
        mylogger.info("Done ! ")
