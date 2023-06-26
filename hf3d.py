# -*- coding: utf-8 -*-

import math
import os

import folium
import matplotlib as mpl
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from haversine import haversine
from scipy.optimize import least_squares, minimize
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder
from streamlit_folium import st_folium


def convert_vector_to_bearing_inclination(vector):
    # Convert a vector to a bearing and inclination
    # NEV와 xyz 관계는 y축 기준 180도 회전
    # N축은 -1,0,0
    x, y, z = -vector[0], vector[1], -vector[2]
    bearing = int(round(np.degrees(math.acos(np.dot([-1, 0, 0], [x, y, 0])/(1*np.sqrt(x**2+y**2)))),0))
    incl = int(round(np.degrees(math.acos(np.dot([x, y, 0], [x, y, z])/(np.sqrt(x**2+y**2+z**2)*np.sqrt(x**2+y**2)))),0)    )
    if z <= 0:
        incl = abs(incl)
    else:
        incl = -abs(incl)
    if x >=0 and y>=0:
        bear = f"S{180-bearing}E"
    elif x >=0 and y<0:
        bear = f"S{180-bearing}W"
    elif x <0 and y>=0:
        bear = f"N{bearing}E"
    elif x <0 and y<0:
        bear = f"N{bearing}W"
    return bear, incl
    
    

def anno_func(x):
    if x > 0:
        return f"-{abs(x):0.3f}"
    else:
        return f"+{abs(x):0.3f}"

def chart_Option(FIG, x_axis_min, x_axis_max, anno):
        FIG.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        FIG.update_yaxes(
            autorange="reversed",
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGrey",
            mirror=True,
            title_font=dict(size=22),
            tickfont=dict(size=18),
        )

        FIG.update_xaxes(
            range=[x_axis_min * 1.2, x_axis_max * 1.2],
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGrey",
            mirror=True,
            title_font=dict(size=22),
            tickfont=dict(size=18),
        )

        FIG.update_layout(
            template="simple_white",
            font=dict(size=24,color="black"),
            legend_title=dict(font=dict(size=20)),
            legend=dict(font=dict(size=18)),
            margin=dict(l=100, r=350, t=50, b=50),
        )    

        if anno != "":
            FIG.add_annotation(
            text=anno,
            font=dict(size=18),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.03,
            y=0.75,
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=0,
        )

    
def fs(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.sin(math.radians(ldeg))**lnn, deg, nnn)))

def fc(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.cos(math.radians(ldeg))**lnn, deg, nnn)))


def calMF(x0, data):
# data: findex, den, depth, burden, bbering, binclin, mdepth, psm, fstrike, fdip
       
    global psc_final, fSN, fSE, fSV, fSNE, fSEV, fSVN

    fractype, den, tdepth, over = [data[:,i] for i in range(0,4)]
    alpha, beta, dep, psm, psi, pi = [data[:,i] for i in range(4,10)]

    [SN0, SE0, SV0, SNE0, SEV0, SVN0, alphaNN, alphaEE] = x0
    
    depth = dep - tdepth
    alphaVV = den
    alphaNE = 0.5*(alphaNN-alphaEE)*2.*SNE0/(SN0-SE0)
    alphaEV = 0.5*(alphaEE-alphaVV)*2.*SEV0/(SE0-SV0)
    alphaVN = 0.5*(alphaVV-alphaNN)*2.*SVN0/(SV0-SN0)

    SN = SN0 + depth*alphaNN
    SE = SE0 + depth*alphaEE
    SV = SV0 + depth*alphaVV
    SNE = SNE0 + depth*alphaNE
    SEV = SEV0 + depth*alphaEV
    SVN = SVN0 + depth*alphaVN

    
    ver = tuple([fractype == 0])
    inc = tuple([fractype > 0])

    # Synn 
    psc = SN*fc(2., pi)*fc(2., psi) + SE*fc(2., pi)*fs(2., psi) + SV*fs(2., pi) + SNE*fc(2., pi)*fs(1., 2.*psi) + SEV*fs(1., 2.*pi)*fs(1., psi) + SVN*fs(1., 2.*pi)*fc(1.,psi)
    # Mizuta  - non-correct
    #psc = SN*fc(2., pi)*fc(2., psi) + SE*fc(2., pi)*fc(2., psi) + SV*fs(2., pi) + SNE*fc(2., pi)*fs(1., 2.*psi) + SEV*fs(1., 2.*pi)*fs(1., psi) + SVN*fs(1., 2.*pi)*fc(1.,psi)

    psc_final = psc
    
    fSN, fSE, fSV, fSNE, fSEV, fSVN = [SN, SE, SV, SNE, SEV, SVN]

    errsum = (psm-psc)**2. + (fSV-(over+den*depth))**2.
    errsum = (errsum.sum()/(len(errsum)-1))**.5

    return errsum


@st.cache_data()
def BH_and_wsm_func(WSM_file, lat, lng):    
    '''
    wsm 파일명, 위도, 경도 정보를 수신하여 wsm 파일에서
    해당 위도, 경도에 가장 가까운 측정소 5개를 찾아 표시
    '''
       
    wsm = pd.read_csv(WSM_file, encoding='ISO-8859-1', engine='python')
    kor = wsm[wsm['COUNTRY']=='Korea - Republic of']
    
    kor['dist_wsm_src(km)'] = list(map(lambda slat,slng: round(haversine((lat,lng),(slat,slng), unit='km'),2), kor['LAT'], kor['LON']))
    kor_sorted = kor.sort_values(by=['dist_wsm_src(km)'])[
        ['ID','dist_wsm_src(km)','LAT','LON','TYPE','DEPTH','QUALITY','REGIME','LOCALITY','DATE','NUMBER',
         'SD','METHOD','S1AZ','S1PL','S2AZ','S2PL','S3AZ','S3PL','MAG_INT_S1','SLOPES1','MAG_INT_S2','SLOPES2','MAG_INT_S3','SLOPES3']]
        
    return kor_sorted[0:5], kor_sorted[5::]
    
def ini_reload():
    st.session_state.reload = True

# main() Function
if __name__ == "__main__": 
    
    
    st.set_page_config(
        page_title="HF3Dpy",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    api_key = os.environ.get('API_KEY_KIGAM')
            
    tab_title_list = ["평면도", "데이터처리결과", "측정값과 계산값 간 차이", "심도별 응력", "심도별 주응력", "주응력 방향 분포"]
    
    WSM_file = '.\wsm2016.csv'

    st.sidebar.markdown("## HF3Dpy")
    st.sidebar.markdown(f"지하공간 형태나 균열대 분포에 따라 경사시추공을 이용한 삼차원 초기지압 측정이 필요할 때 본 코드를 이용하여 현지 측정값과 계산값 간의 오차를 최소화함으로써 응력을 추정함.")
    st.sidebar.markdown("**참고자료**")
    st.sidebar.markdown(f"[이론적 배경 (Synn et al., 2015)](http://dx.doi.org/10.1016/j.ijrmms.2015.01.012)")
    st.sidebar.markdown(f"[최적화 함수(BFGS)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)")
    st.sidebar.markdown(f"[World Stress Map 안내서 및 데이터 파일](https://datapub.gfz-potsdam.de/download/10.5880.WSM.2016.001/)")
    
    st.sidebar.markdown(f"---")

    
    
    dfile = st.sidebar.file_uploader(label="**데이터 파일 선택**", type=['dat', 'txt'], on_change=ini_reload())
           
    if 'export_disabled' not in st.session_state:
        st.session_state.export_disabled = True
    
    if 'remark' not in st.session_state:
        st.session_state.remark = ''
    
    if 'remark2' not in st.session_state:
        st.session_state.remark2 = ''

    if 'reload' not in st.session_state:
        st.session_state.reload = True
    
    st.sidebar.markdown("---")        
    st.sidebar.markdown(st.session_state.remark)
    st.sidebar.markdown(st.session_state.remark2)


    title_layer = st.container()
   
    if dfile is None:
        st.session_state.remark = ''
        st.session_state.remark2 = ''
        
    elif dfile.name[-3:].lower()=="dat" or dfile.name[-3:].lower()=="txt":
            
        infofile = dfile.name[:-4]+"_info.txt"

        if os.path.isfile(infofile):
            st.session_state.remark = '관련 info 파일 있음.'
            st.session_state.remark2 = '그래프, 표 png, csv 파일로 저장 가능(지도 제외).'
                
            try:
                # info 파일 읽기
                f = open(infofile, mode='r', encoding='utf-8')
                info = f.readlines()
                test = info[1].replace('\n','')
                project = info[3].replace('\n','')
                date = pd.to_datetime(info[5].replace('\n',''))
                lat = pd.to_numeric(info[7].replace('\n','').split(",")[0])
                lng = pd.to_numeric(info[7].replace('\n','').split(",")[1])
                
                # 공통 정보 표시
                with title_layer:
                    if test != '':
                        st.markdown(f"## {test}")
                    if project != '':    
                        st.markdown(f"### {project}")
                    if date:
                        st.markdown(f"- 시험일: {date:%Y-%m-%d}")
                    if lat != 0:
                        st.markdown(f"- 위치(위도, 경도): {lat}, {lng}")
                            
                        
                # 평면도 정보 처리
                        
                # 시추공 위경도 중심으로 인접한 5개 WSM 데이터 표시
                # wsm file, lat, lng 전달
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_title_list)
                with tab1:
                    with st.spinner(f"평면도 생성 중"):
                        m = folium.Map(location=[lat, lng], zoom_start=10)
                           
                        #layers = 'L_50K_Geology_Map'
                        #folium.WmsTileLayer('https://data.kigam.re.kr/openapi/wms', layers, styles='', FORMAT='image/png', transparent=True, version='1.1.1', attr='', name="5만 지질도", overlay=True, control=True, show=False, key=api_key).add_to(m)
                        #layers = 'G_tectonic'
                        #folium.WmsTileLayer('https://data.kigam.re.kr/openapi/wms', layers, styles='', FORMAT='image/png', transparent=True, version='1.1.1', attr='', name="지체구조도", overlay=True, control=True, show=True, key=api_key).add_to(m)
                        #folium.LayerControl().add_to(m)
                            
                        folium.Marker([lat,lng], popup=test,tooltip=test).add_to(m)

                        wsm_near_five, wsm_others = BH_and_wsm_func(WSM_file,lat,lng)
                                
                        #wsm_near_five.apply(lambda row:folium.CircleMarker([row['LAT'],row['LON']], popup=row['ID'], tooltip=row['ID'], radius=8, color='red', fill='blue').add_to(m), axis=1)
                        #wsm_others.apply(lambda row:folium.CircleMarker([row['LAT'],row['LON']], popup=row['ID'], tooltip=row['ID'], radius=6, color='black', fill='blue').add_to(m), axis=1)
            
                        st_folium(m, width=1200, height=600)
                                
                        st.markdown(f"측정지점과 가장 가까운 5개의 WSM 데이터")
                        AgGrid(wsm_near_five, columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS, udate_on = ['init'])                 
                        st.caption("자료 항목에 대한 자세한 설명은 참고자료 \'WSM 안내서\'에서 확인 가능함.")

            except:
                st.session_state.remark = '관련 info 파일 읽기 오류. 데이터만 처리함.'
                st.session_state.remark2 = '그래프, 표 png, csv 파일로 저장 가능(지도 제외).'
                tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_title_list[1::])
        else:
            st.session_state.remark = "관련 info 파일 없음. 데이터만 처리함."
            st.session_state.remark2 = '그래프, 표 png, csv 파일로 저장 가능(지도 제외).'        
            tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_title_list[1::])


        #### data processing ####
        # reading dat file
        src = dfile.read().decode('utf-8')
        src = src.split('\r')

        # line 1-3
        density, tdepth, tburden = [
            float(xx) for xx in src[0].replace("\n", "").split('\t')[0:3]]
        x0 = [float(xx)
              for xx in src[1].replace("\n", "").split('\t')]
        norows = int(src[2].replace("\n", "").split('\t')[0])

        # line 4-n 
        temp = list()
        for ii in range(0, norows, 1):
            temp.append(src[ii+3].replace("\n", "").split('\t'))
        m = pd.DataFrame(temp, columns=['findex', 'bbering', 'binclin', 'mdepth', 'psm', 'fstrike', 'fdip', 'dummy'])

        m.insert(1, 'cden', density)
        m.insert(2, 'cz0', tdepth)
        m.insert(3, 'cburden', tburden)
        m.drop('dummy', axis=1, inplace=True)
        m = m.astype(float)
        m['findex'] = m['findex'].astype(int)
        data = m.to_numpy()

        
        result = minimize(calMF, x0, data, tol=1.e-5, method='BFGS')        
        

        fi, mdep, cz0, psm, cburden, cden = data[:,0], data[:,6], data[:,2], data[:,7], data[:,3], data[:,1]
        df = pd.DataFrame([fi, mdep, cz0, mdep-cz0, psc_final, psm,
                          psm-psc_final, fSV-(cburden+cden*(mdep-cz0)), fSN, fSE, fSV, fSNE, fSEV, fSVN])
        df = df.T
        df.columns = ["Fracture_type", "mdepth", "tdepth", "depth", "Psc", "Psm",
                      "tolPs", "tolPv", "PN", "PE", "PV", "PNE", "PEV", "PVN"]        

        df = df.round(3)
        df["Fracture_type"] = df["Fracture_type"].astype(int)


        # ---- Stereonet -------
        mag = [[] for i in range(3)]
        vec = [[] for i in range(3)]
        for idx, [PN, PE, PV, PNE, PEV, PVN] in df[["PN", "PE", "PV", "PNE", "PEV", "PVN"]].iterrows():
            sigma = np.asarray([[PN, PNE, PVN],
                                [PNE, PE, PEV],
                                [PVN, PEV, PV]])

            e_val, e_vec = np.linalg.eig(sigma)
            # sort by magnitude of eigenvalues
            idx = e_val.argsort()[::-1]
            e_val = e_val[idx]
            e_vec = e_vec[:, idx]
            for ii in range(0, 3):
                mag[ii].append(np.round(e_val[ii],4))
                vec[ii].append(np.round(e_vec[ii],4))

        df["P1mag"] = mag[0]
        df["P2mag"] = mag[1]
        df["P3mag"] = mag[2]
        df["P1vec"] = vec[0]
        df["P2vec"] = vec[1]
        df["P3vec"] = vec[2]
        
        t1 = pd.DataFrame([convert_vector_to_bearing_inclination(xx) for xx in df["P1vec"]])
        df['P1_bear.'] = t1.iloc[:][0]
        df['P1_incl.'] = t1.iloc[:][1]
        t2 = pd.DataFrame([convert_vector_to_bearing_inclination(xx) for xx in df["P2vec"]])
        df['P2_bear.'] = t2.iloc[:][0]
        df['P2_incl.'] = t2.iloc[:][1]
        t3 = pd.DataFrame([convert_vector_to_bearing_inclination(xx) for xx in df["P3vec"]])
        df['P3_bear.'] = t3.iloc[:][0]
        df['P3_incl.'] = t3.iloc[:][1]
        
        
        x1 = [xx[0] for xx in df["P1vec"]]
        y1 = [xx[1] for xx in df["P1vec"]]
        z1 = [xx[2] for xx in df["P1vec"]]
        x2 = [xx[0] for xx in df["P2vec"]]
        y2 = [xx[1] for xx in df["P2vec"]]
        z2 = [xx[2] for xx in df["P2vec"]]
        x3 = [xx[0] for xx in df["P3vec"]]
        y3 = [xx[1] for xx in df["P3vec"]]
        z3 = [xx[2] for xx in df["P3vec"]]            


        # ---- Plotly -------
        ## fig 1
        gf1 = pd.concat([df.tolPs, df.mdepth], axis=1)
        gf1.rename(columns={"tolPs": "X"}, inplace=True)
        gf1["Legend"] = "Psm-Psc"
        gf2 = pd.concat([df.tolPv, df.mdepth], axis=1)
        gf2.rename(columns={"tolPv": "X"}, inplace=True)
        gf2["Legend"] = "PN-(tburden+depth*den)"
        gf = pd.concat([gf1, gf2])
        fig1 = px.scatter(
            gf,
            x="X",
            y="mdepth",
            color="Legend",
            labels=dict(
                X="Psm-Psc or PN-(tburden+depth*den) (MPa)",
                mdepth="Depth (m)",
                Legend="Stress",
            ),
            height=800, width=1000,
        )
        
        x_axis_max = max(max(abs(df.tolPs)), max(abs(df.tolPv)))
        x_axis_min = -x_axis_max
        
        chart_Option(fig1, x_axis_min, x_axis_max, anno="")


        ## fig 2
        gf1 = pd.concat([df.PN, df.mdepth], axis=1)
        gf1.rename(columns={"PN": "X"}, inplace=True)
        gf1["Legend"] = "PN"

        gf2 = pd.concat([df.PE, df.mdepth], axis=1)
        gf2.rename(columns={"PE": "X"}, inplace=True)
        gf2["Legend"] = "PE"

        gf3 = pd.concat([df.PV, df.mdepth], axis=1)
        gf3.rename(columns={"PV": "X"}, inplace=True)
        gf3["Legend"] = "PV"

        gfA = pd.concat([gf1, gf2, gf3])

        fig2 = px.scatter(
            gfA,
            x="X",
            y="mdepth",
            color="Legend",
            trendline="ols",
            labels=dict(X="PN or PE or PV (MPa)", mdepth="Depth (m)", Legend="Stress"),
            height=800, width=1000,
        )

        PNEV_results = px.get_trendline_results(fig2).px_fit_results.iloc[0:3]
        
        x_axis_max = max(
            max(df.PN), max(df.PE), max(df.PV), max(df.PNE), max(df.PEV), max(df.PVN)
        )
        x_axis_min = min(
            0,
            min(min(df.PN), min(df.PE), min(df.PV)),
            min(min(df.PNE), min(df.PEV), min(df.PVN)),
        )
        
        anno = f"PN={1./PNEV_results[0].params[1]:0.3f}*Depth{anno_func(PNEV_results[0].params[0]/PNEV_results[0].params[1])} (r2={PNEV_results[0].rsquared:0.2f})<br>"
        anno = anno + f"PE={1./PNEV_results[1].params[1]:0.3f}*Depth{anno_func(PNEV_results[1].params[0]/PNEV_results[1].params[1])} (r2={PNEV_results[1].rsquared:0.2f})<br>"
        anno = anno + f"PV={1./PNEV_results[2].params[1]:0.3f}*Depth{anno_func(PNEV_results[2].params[0]/PNEV_results[2].params[1])} (r2={PNEV_results[2].rsquared:0.2f})<br>"
        
        chart_Option(fig2, x_axis_min, x_axis_max, anno)
        
        

        gf4 = pd.concat([df.PNE, df.mdepth], axis=1)
        gf4.rename(columns={"PNE": "X"}, inplace=True)
        gf4["Legend"] = "PNE"

        gf5 = pd.concat([df.PEV, df.mdepth], axis=1)
        gf5.rename(columns={"PEV": "X"}, inplace=True)
        gf5["Legend"] = "PEV"

        gf6 = pd.concat([df.PVN, df.mdepth], axis=1)
        gf6.rename(columns={"PVN": "X"}, inplace=True)
        gf6["Legend"] = "PVN"

        gfB = pd.concat([gf4, gf5, gf6])

        fig4 = px.scatter(
            gfB,
            x="X",
            y="mdepth",
            color="Legend",
            trendline="ols",
            labels=dict(X="PNE or PEV or PVN (MPa)", mdepth="Depth (m)", Legend="Stress"),
            height=800, width=1000,
        )

        PNEV_results = px.get_trendline_results(fig4).px_fit_results.iloc[0:3]

        anno = f"PNE={1./PNEV_results[0].params[1]:0.3f}*Depth{anno_func(PNEV_results[0].params[0]/PNEV_results[0].params[1])} (r2={PNEV_results[0].rsquared:0.2f})<br>"
        anno = anno + f"PEV={1./PNEV_results[1].params[1]:0.3f}*Depth{anno_func(PNEV_results[1].params[0]/PNEV_results[1].params[1])} (r2={PNEV_results[1].rsquared:0.2f})<br>"
        anno = anno + f"PVN={1./PNEV_results[2].params[1]:0.3f}*Depth{anno_func(PNEV_results[2].params[0]/PNEV_results[2].params[1])} (r2={PNEV_results[2].rsquared:0.2f})<br>"
        
        chart_Option(fig4, x_axis_min, x_axis_max, anno)
        
        #fig6 = ff.create_quiver(0,0,0,-1,0,0,)
        
        fig6, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'), figsize=(10,10))
                          
        plt.rcParams["grid.linewidth"] = 0.3
        plt.tight_layout()
                    
        views = [[(20,15,0), (90,0,0)],
                 [(0,0,0), (0,90,0)]]
        
        for i in range(2):
            for j in range(2):
              
                ax[i][j].xaxis.set_ticklabels([])
                ax[i][j].yaxis.set_ticklabels([])
                ax[i][j].zaxis.set_ticklabels([])
                #ax[i][j].xaxis.set_ticks([])
                #ax[i][j].yaxis.set_ticks([])
                #ax[i][j].zaxis.set_ticks([])

                # Make data
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = 1 * np.outer(np.cos(u), np.sin(v))
                y = 1 * np.outer(np.sin(u), np.sin(v))
                z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))


                # Plot the surface
                #ax[i][j].plot_surface(x, y, z, color='gray', alpha=0.1)

                ax[i][j].quiver(0,0,0,-1,0,0,color='k', label=None, arrow_length_ratio=0.1)
                ax[i][j].quiver(0,0,0,0,1,0,color='k', label=None, arrow_length_ratio=0.1)
                ax[i][j].quiver(0,0,0,0,0,-1,color='k', label=None, arrow_length_ratio=0.1)

                ax[i][j].text(-1.05, 0, 0.05, 'N', None)        
                ax[i][j].text(0, 1.05, 0.05, 'E', None)        
                ax[i][j].text(0, 0, -1.1, 'V', None)
                ax[i][j].set_xlim([-1.2,1.2])        
                ax[i][j].set_ylim([-1.2,1.2])        
                ax[i][j].set_zlim([-1.2,1.2])        
                #ax[i][j].set_xlabel('N')
                #ax[i][j].set_ylabel('E')
                #ax[i][j].set_zlabel('V')

                for idx, [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]] in df[["P1vec", "P2vec", "P3vec"]].iterrows():
                    ax[i][j].quiver(0,0,0,-x1, y1, -z1, color='r', label='P1' if idx==0 else None, arrow_length_ratio=0.15)
                    ax[i][j].quiver(0,0,0,-x2, y2, -z2, color='g', label='P2' if idx==0 else None, arrow_length_ratio=0.15) 
                    ax[i][j].quiver(0,0,0,-x3, y3, -z3, color='b', label='P3' if idx==0 else None, arrow_length_ratio=0.15)
                    ax[i][j].set_proj_type('ortho')
                    ax[i][j].set_box_aspect([1,1,1], zoom=0.9)
                    ax[i][j].view_init(views[i][j][0], views[i][j][1], views[i][j][2])

        ax[0][0].legend(loc='best')
        ax[0][1].set_title('Plan view')
        ax[1][0].set_title('Front view')
        ax[1][1].set_title('Side view')
        
        
        #ax1 = fig6.add_subplot(221, projection="stereonet")
        #ax2 = fig6.add_subplot(223, projection="stereonet")
        #ax3 = fig6.add_subplot(224, projection="stereonet")
#
        #ax1.set_azimuth_ticks([])
        #ax2.set_azimuth_ticks([])
        #ax3.set_azimuth_ticks([])
#
#
        #plunge1, bearing1 = mplstereonet.vector2plunge_bearing(x1, y1, z1)
        #plunge2, bearing2 = mplstereonet.vector2plunge_bearing(x2, y2, z2)
        #plunge3, bearing3 = mplstereonet.vector2plunge_bearing(x3, y3, z3)
        #strike1, dip1 = mplstereonet.vector2pole(x1, y1, z1)
#
        ## Make a density contour plot of the orientations
        #ax1.density_contourf(plunge1, bearing1, measurement="lines")
        #ax1.line(plunge1, bearing1, marker="o", color="black")
        #ax1.grid(True)
        #ax1.set_title("Major Principal Stress", font=dict(size=9))
        ##ax1.set_azimuth_ticks(range(0, 360, 10))
#
        #ax2.density_contourf(plunge2, bearing2, measurement="lines")
        #ax2.line(plunge2, bearing2, marker="o", color="black")
        #ax2.grid(True)
        #ax2.set_title("Intermediate Principal Stress", font=dict(size=9))
        ##ax2.set_azimuth_ticks(range(0, 360, 10))
#
        #ax3.density_contourf(plunge3, bearing3, measurement="lines")
        #ax3.line(plunge3, bearing3, marker="o", color="black")
        #ax3.grid(True)
        #ax3.set_title("Minor Principal Stress", font=dict(size=9))
        ##ax3.set_azimuth_ticks(range(0, 360, 10))


        # ---------
        gf1 = pd.concat([df.P1mag, df.mdepth], axis=1)
        gf1.rename(columns={"P1mag": "X"}, inplace=True)
        gf1["Legend"] = "P1"

        gf2 = pd.concat([df.P2mag, df.mdepth], axis=1)
        gf2.rename(columns={"P2mag": "X"}, inplace=True)
        gf2["Legend"] = "P2"

        gf3 = pd.concat([df.P3mag, df.mdepth], axis=1)
        gf3.rename(columns={"P3mag": "X"}, inplace=True)
        gf3["Legend"] = "P3"

        gfA = pd.concat([gf1, gf2, gf3])

        fig7 = px.scatter(
            gfA,
            x="X",
            y="mdepth",
            color="Legend",
            trendline="ols",
            labels=dict(
                X="Principal stress (MPa)", mdepth="Depth (m)", Legend="Principal stress"
            ),
            height=800, width=1000,
        )

        PRINCIPAL_results = px.get_trendline_results(fig7).px_fit_results.iloc[0:3]

        anno = f"P1={1./PRINCIPAL_results[0].params[1]:0.3f}*Depth{anno_func(PRINCIPAL_results[0].params[0]/PRINCIPAL_results[0].params[1])} (r2={PRINCIPAL_results[0].rsquared:0.2f})<br>"
        anno = anno + f"P2={1./PRINCIPAL_results[1].params[1]:0.3f}*Depth{anno_func(PRINCIPAL_results[1].params[0]/PRINCIPAL_results[1].params[1])} (r2={PRINCIPAL_results[1].rsquared:0.2f})<br>"
        anno = anno + f"P3={1./PRINCIPAL_results[2].params[1]:0.3f}*Depth{anno_func(PRINCIPAL_results[2].params[0]/PRINCIPAL_results[2].params[1])} (r2={PRINCIPAL_results[2].rsquared:0.2f})<br>"
        
        chart_Option(fig7, x_axis_min, x_axis_max, anno)
        

        # ---------
        df["P1_P2"] = df.P1mag / df.P2mag
        df["P1_P3"] = df.P1mag / df.P3mag
        df["P2_P3"] = df.P2mag / df.P3mag
        gf1 = pd.concat([df.P1_P2, df.mdepth], axis=1)
        gf1.rename(columns={"P1_P2": "X"}, inplace=True)
        gf1["Legend"] = "P1/P2"

        gf2 = pd.concat([df.P2_P3, df.mdepth], axis=1)
        gf2.rename(columns={"P2_P3": "X"}, inplace=True)
        gf2["Legend"] = "P2/P3"

        gf3 = pd.concat([df.P1_P3, df.mdepth], axis=1)
        gf3.rename(columns={"P1_P3": "X"}, inplace=True)
        gf3["Legend"] = "P1/P3"

        gfA = pd.concat([gf1, gf2, gf3])

        fig8 = px.scatter(
            gfA,
            x="X",
            y="mdepth",
            color="Legend",
            trendline="ols",
            labels=dict(X="Principal stress ratio", mdepth="Depth (m)", Legend="Legend"),
            height=800, width=1000,
        )

        PR_results = px.get_trendline_results(fig8).px_fit_results.iloc[0:3]

        anno = f"P1/P2={1./PR_results[0].params[1]:0.3f}*Depth{anno_func(PR_results[0].params[0]/PR_results[0].params[1])} (r2={PR_results[0].rsquared:0.2f})<br>"
        anno = anno + f"P2/P3={1./PR_results[1].params[1]:0.3f}*Depth{anno_func(PR_results[1].params[0]/PR_results[1].params[1])} (r2={PR_results[1].rsquared:0.2f})<br>"
        anno = anno + f"P1/P3={1./PR_results[2].params[1]:0.3f}*Depth{anno_func(PR_results[2].params[0]/PR_results[2].params[1])} (r2={PR_results[2].rsquared:0.2f})<br>"
        
        chart_Option(fig8, 0, np.max(df['P1_P3']), anno)

        
        with tab2:
            st.markdown("### 데이터처리 결과")
            st.markdown(f"Rock density (kN/m3): {density*1000.}")
            st.markdown(f"Total depth from ground surface to upper boundary of bedrock (m): {tdepth}")
            st.markdown(f"Vertical stress of overburden (MPa): {tburden}")            

            AgGrid(df, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, update_on = 'init',
                   default_column_parameters={'editable': False, 'sortable': True, 'filter': True, 'resizable': True, 'minWidth': 100, 
                                            'flex': 1, 'floatingFilter': True, 'cellStyle': {'textAlign': 'right'}})
                                            

        with tab3:
            st.plotly_chart(fig1)
        with tab4:
            st.plotly_chart(fig2)
            st.plotly_chart(fig4)
        with tab5:
            st.plotly_chart(fig7) 
            st.plotly_chart(fig8) 
        with tab6:
            col1, col2,col3 = st.columns([1,1,0.1])
            with col1:
                   AgGrid(df[['P1_bear.','P1_incl.','P2_bear.','P2_incl.','P3_bear.','P3_incl.']], columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, update_on=['init', 'gridChanged', 'viewportChanged'])
            with col2:
                placeholder = st.empty()
                placeholder.pyplot(fig6)
        
        st.session_state.reload = False
