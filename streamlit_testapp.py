#import libs
import pandas as pd
import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
import streamlit as st
from datetime import datetime

import matplotlib.pyplot as plt
import time

####################################
# 1_ Define Functions
# - Load data
# - Clean data
####################################




####################################
# 2_ Engineer Data
# - prepare data for output
####################################




####################################
# 3_ Build Dashboard
####################################


DIVIDER = '--------------------------------------------------------------------------------'


####################################
# - Sidebar
####################################
with st.sidebar:
    #st.image("http://hcc.snu.ac.kr/wordpress/wp-content/uploads/2016/03/lab-logo-1x-e1462202258209.png", width=300)
    st.image("https://hcclab.snu.ac.kr:12121/apps/theming/image/logo?useSvg=1&v=7", width=300)

VIEW_WELCOME = 'Welcome'

VIEW_API_TEXT = 'Text elements & st.write'
VIEW_API_DATA = 'Data display elements'
VIEW_API_CHART = 'Chart elements'
VIEW_UBER_NYC = 'Sample: Uber picksup in NYC'



sidebar = [VIEW_WELCOME, VIEW_API_TEXT,VIEW_API_DATA,VIEW_API_CHART,VIEW_UBER_NYC]
add_sidebar = st.sidebar.selectbox('Selectbox below', sidebar) #('Aggregate Metrics','Individual Video Analysis'))


with st.sidebar:
    add_radio = st.radio(
        "Choose a language (not working)",
        ("English", "Korean")
    )



####################################
# - VIEW_WELCOME
####################################


if add_sidebar == VIEW_WELCOME:
    st.title(add_sidebar)
    st.header('Let\'s learn Streamlit library')


    st.write(DIVIDER)
    st.subheader('References')
    url1 = 'https://docs.streamlit.io/'
    url2 = 'https://bittersweet-match-49f.notion.site/Streamlit-5ca73e87f96a443a902eefc5c721e3d0'

    url1
    url2


    st.write(DIVIDER)
    name = st.text_input('Name')
    if not name:
        st.warning('Please input a name.')
        st.stop()
    st.success('Thank you for inputting a name.')

    st.write(DIVIDER)
    st.subheader('Template')    

    code = '''
        #<Sample code for the template>
        
        import streamlit as st
        import pandas as pd
        
        ####################################
        # 1_ Define Functions
        # - Load data
        # - Clean data
        ####################################

        @st.cache
        def load_data():
            #filepath = 'some_filepath.csv'
            #df = pd.read_csv(filepath)
            df = pd.DataFrame({
                'first column': [1, 2, 3, 4],
                'second column': [10, 20, 30, 50],
            })
            return df

        df = load_data()

        ####################################
        # 2_ Engineer Data
        # - prepare data for output
        ####################################

        df.column = ['Number', 'Scores']
        df['Level'] = ['C', 'B', 'B', 'A']

        ####################################
        # 3_Build Dashboard using Streamlit
        ####################################

        st.title('You can build Streamlit Webapp')
        add_sidebar = st.sidebar.selectbox('Select Method', 'First','Second')

        if add_sidebar == 'First':
            # View for s'First'- show matrix, graph etc.
        else :
            # View for 'Second' - show matrix, graph etc.
        '''
    st.code(code, language='python')

    # with st.spinner("Checking the connection..."):
    #     time.sleep(2)
    #     st.success("You are in perfect conditioin.")



####################################
# - VIEW_API_TEXT
####################################

elif add_sidebar == VIEW_API_TEXT:
    st.title(add_sidebar)

    
    with st.container():
        #st.write(f'(1) Text Elements')

        st.header('st.header | (1) Text Elements')
        st.subheader('st.subheader | texts for formatted texts')

        st.markdown('st.markdown | Streamlit is **_really_ cool**.')
        st.caption('st.caption | caption: This is a string that explains something above.')


        code = '''
            def hello():
            print("This is code block for python")'''
        st.code(code, language='python')

        code = '''
            <script type="text/javascript">
            alert("Hello Javatpoint");
            </script>'''
        st.code(code, language='javascript')


        with st.echo():
            st.write("with st.echo(): ")
            st.write("This is another way to show code.")



        st.text('st.text | This is some text.')

        st.latex(r'''
            st.latex = a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
            \sum_{k=0}^{n-1} ar^k =
            a \left(\frac{1-r^{n}}{1-r}\right)
            ''')


    st.write(DIVIDER)


    with st.container():
        #st.write(f'(2) st.write and Magic function')
        st.header('(2) st.write and Magic')
        st.subheader('(2-1) st.write can write many things')

        st.write(1234)
        st.write('1000+ 4, ' , (1000+4))
        st.write('Below is a pandas dataframe')
        st.write(pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 50],
        }))

        st.subheader('(2-2) Magic write without command')
        tab_names = ['Valuable','Pandas Dataframe', 'Matplotlib']
        tab1, tab2, tab3 = st.tabs(tab_names)

        with tab1:
            st.header(tab_names[0])
            x = 10
            y = 20
            'x', x  #  Draw the string 'x' and then the value of x
            'y', y
            'x+y', x+y


        with tab2:
            st.header(tab_names[1])
            test_df = pd.DataFrame({'col1': [1,2,3]})
            test_df  #  Draw the dataframe            

        with tab3:
            st.header(tab_names[2])
            # Also works with most supported chart types
            arr = np.random.normal(1, 1, size=100)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=20)
            fig  #  Draw a Matplotlib chart


####################################
# - \VIEW_UBER_NYC
####################################


elif add_sidebar == VIEW_UBER_NYC:
    st.title(VIEW_UBER_NYC)
	
    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
                'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
        
    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data
        
    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text("Done! (using st.cache)")
        
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
        
    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)
        
    hour_to_filter = st.slider('hour', 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
        
    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.map(filtered_data)



####################################
# - VIEW_API_DATA
####################################

elif add_sidebar == VIEW_API_DATA:
    st.title(add_sidebar)


    with st.echo():
        df = pd.DataFrame(
        np.random.randn(20, 20),
        columns=('col %d' % i for i in range(20)))


    with st.container():
        st.subheader('(1) st.dataframe(pd.dataframe)')
        st.dataframe(df)  # Same as st.write(df)
    
    with st.container():
        st.subheader('(2) st.table(pd.dataframe)')
        st.table(df)    

    with st.container():
        st.subheader('(3) st.metric')
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", "70 °F", "1.2 °F")
        col2.metric("Wind", "9 mph", "-8%")
        col3.metric("Humidity", "86%", "4%")

    with st.container():
        st.json({
            'foo': 'bar',
            'baz': 'boz',
            'stuff': [
                'stuff 1',
                'stuff 2',
                'stuff 3',
                'stuff 5',
            ],
        })
        



####################################
# - VIEW_API_CHART
####################################

elif add_sidebar == VIEW_API_CHART:
    st.title(add_sidebar)

    st.header('(1) Dataframe to streamlit chart')
    with st.echo():
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])

    with st.container():
        st.subheader('(1-1) st.line_chart(chart_data)')
        st.line_chart(chart_data)

    with st.container():
        st.subheader('(1-2) st.area_chart(chart_data)')
        st.area_chart(chart_data)

    with st.container():
        st.subheader('(1-3) st.bar_chart(chart_data)')
        st.bar_chart(chart_data)

    st.write(DIVIDER)
    with st.container():
        st.subheader('(1-4) st.map(df)')
        with st.echo():
            df = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
                columns=['lat', 'lon'])
        st.map(df)


    st.write(DIVIDER)
    st.write(DIVIDER)

    st.header('(2) Python library chart')

    with st.container():
        st.subheader('(2-1) matplotlib.pyplot --> st.pyplot')
        import matplotlib.pyplot as plt
        arr = np.random.normal(1, 1, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        st.pyplot(fig)


    st.write(DIVIDER)

    with st.echo():
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])

    with st.container():
        st.subheader('(2-2) altair.Chart --> st.altair_chart')
        import altair as alt
        df = chart_data
        c = alt.Chart(df).mark_circle().encode(
            x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        st.altair_chart(c, use_container_width=True)


    with st.container():
        st.subheader('(2-3) vega-Lite --> st.vega_lite_chart')
        import pandas as pd
        import numpy as np

        df = chart_data
        st.vega_lite_chart(df, {
            'mark': {'type': 'circle', 'tooltip': True},
            'encoding': {
                'x': {'field': 'a', 'type': 'quantitative'},
                'y': {'field': 'b', 'type': 'quantitative'},
                'size': {'field': 'c', 'type': 'quantitative'},
                'color': {'field': 'c', 'type': 'quantitative'},
            },
        })


    st.write(DIVIDER)
    with st.container():
        st.subheader('(2-4) Plotly --> st.plotly_chart')

        # import plotly.figure_factory as ff
        # import numpy as np

        # with st.echo():
        #     # Add histogram data
        #     x1 = np.random.randn(200) - 2
        #     x2 = np.random.randn(200)
        #     x3 = np.random.randn(200) + 2

        #     # Group data together
        #     hist_data = [x1, x2, x3]

        #     group_labels = ['Group 1', 'Group 2', 'Group 3']

        #     # Create distplot with custom bin_size
        #     fig = ff.create_distplot(
        #             hist_data, group_labels, bin_size=[.1, .25, .5])

        # # Plot!
        # st.plotly_chart(fig, use_container_width=True)

    st.write(DIVIDER)
    with st.container():
        st.subheader('(2-5) Bokeh --> st.bokeh_chart')

        # import streamlit as st
        # from bokeh.plotting import figure

        # with st.echo():
        #     x = [1, 2, 3, 4, 5]
        #     y = [6, 7, 2, 4, 5]
        #     p = figure(
        #         title='simple line example',
        #         x_axis_label='x',
        #         y_axis_label='y')

        # p.line(x, y, legend_label='Trend', line_width=2)
        # st.bokeh_chart(p, use_container_width=True)

    
    st.write(DIVIDER)
    with st.container():
        st.subheader('(2-6) PyDeck --> st.pydeck_chart')

        # with st.echo():
        #     df = pd.DataFrame(
        #     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        #     columns=['lat', 'lon'])

        # st.pydeck_chart(pdk.Deck(
        #     map_style=None,
        #     initial_view_state=pdk.ViewState(
        #         latitude=37.76,
        #         longitude=-122.4,
        #         zoom=11,
        #         pitch=50,
        #     ),
        #     layers=[
        #         pdk.Layer(
        #         'HexagonLayer',
        #         data=df,
        #         get_position='[lon, lat]',
        #         radius=200,
        #         elevation_scale=4,
        #         elevation_range=[0, 1000],
        #         pickable=True,
        #         extruded=True,
        #         ),
        #         pdk.Layer(
        #             'ScatterplotLayer',
        #             data=df,
        #             get_position='[lon, lat]',
        #             get_color='[200, 30, 0, 160]',
        #             get_radius=200,
        #         ),
        #     ],
        # ))



    st.write(DIVIDER)
    with st.container():
        st.subheader('(2-7) Graphicviz (dagre-d3) --> st.graphicviz_chart')
        # import streamlit as st
        # import graphviz as graphviz

        # with st.echo():
        #     # Create a graphlib graph object
        #     graph = graphviz.Digraph()
        #     graph.edge('run', 'intr')
        #     graph.edge('intr', 'runbl')
        #     graph.edge('runbl', 'run')
        #     graph.edge('run', 'kernel')
        #     graph.edge('kernel', 'zombie')
        #     graph.edge('kernel', 'sleep')
        #     graph.edge('kernel', 'runmem')
        #     graph.edge('sleep', 'swap')
        #     graph.edge('swap', 'runswap')
        #     graph.edge('runswap', 'new')
        #     graph.edge('runswap', 'runmem')
        #     graph.edge('new', 'runmem')
        #     graph.edge('sleep', 'runmem')

        # st.graphviz_chart(graph)

             
