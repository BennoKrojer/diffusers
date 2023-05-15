import pandas as pd 
import plotly.express as px

data = [{'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Flickr Text','score':55.3},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Flickr Text','score':52.7},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Flickr Image','score':46.1},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Flickr Image','score':41.6},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'ARO VG Attr.','score':59.2},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'ARO VG Attr.','score':59.5},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'ARO VG Rel.','score':49.8},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'ARO VG Rel.','score':50.7},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'COCO Order','score':24.8},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'COCO Order','score':23.9},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Flickr Order','score':31.6},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Flickr Order','score':31.0},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'CLEVR','score':65.7},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'CLEVR','score':61.5},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Pets','score':62.5},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Pets','score':63.1},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'SVO Verb','score':71.2},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'SVO Verb','score':68.6},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'SVO Subj','score':74.1},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'SVO Subj','score':72.6},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'SVO Obj','score':79.4},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'SVO Obj','score':76.1},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'ImageCoDe static','score':30.1},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'ImageCoDe static','score':26.0},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'ImageCoDe video','score':15.7},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'ImageCoDe video','score':13.9},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Winoground group','score':7.5},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Winoground group','score':8.3},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Winoground text','score':32.3},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Winoground text','score':29.0},
        {'model':'Diffusion ITM with Stable Diffusion 2.1','task':'Winoground image','score':9.0},
        {'model':'Diffusion ITM with Stable Diffusion 1.5','task':'Winoground image','score':10.7},
        ]

data = [data[i^1] for i in range(len(data))]

# {'Flickr':{'Diffusion ITM with Stable Diffusion 2.1':61.5,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'SVO_verb':{'Diffusion ITM with Stable Diffusion 2.1':77.3,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'SVO_subj':{'Diffusion ITM with Stable Diffusion 2.1':80.5,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'SVO_obj':{'Diffusion ITM with Stable Diffusion 2.1':86.2,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'ImageCoDe_static':{'Diffusion ITM with Stable Diffusion 2.1':42.5,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'ImageCoDe_video':{'Diffusion ITM with Stable Diffusion 2.1':21.1 ,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'Winoground_group':{'Diffusion ITM with Stable Diffusion 2.1':7.7 ,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'Winoground_text':{'Diffusion ITM with Stable Diffusion 2.1':37.5 ,'Diffusion ITM with Stable Diffusion 1.5':0},
#         'Winoground_image':{'Diffusion ITM with Stable Diffusion 2.1':10.2,'Diffusion ITM with Stable Diffusion 1.5':0}
#         }
df = pd.DataFrame(data)

import plotly.express as px

fig = px.bar(df, x="task", y="score", color="model",
             barmode='group',
             labels={
                 "task": "Task",
                 "score": "Accuracy",
                 "model": "Model"
             },
             height=400, width=100, text_auto=True,
             )

# Update the layout properties
fig.update_layout(
    autosize=True,  # Set the autosize option to True
    height=1000,  # Increase the height of the figure
    width=3000,  # Increase the width of the figure
    font=dict(size=40, family="Times New Roman, bold"),
    legend=dict(x=0.72,y=1,bgcolor='rgba(0,0,0,0)',
        font=dict(size=40, family="Times New Roman, bold"),  # Set the font size and family of the legend
    ),  # Set the font size of the text
)
# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', 
#                   insidetextfont=dict(size=25))  # adjust the size as per your need

# Save the figure
fig.write_image("1.5vs2.1.png")
