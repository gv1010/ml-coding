import cv2
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import time

st.set_page_config(page_title="New App", layout="wide")

st.title("ACME Smart Converter")
st.header("Smart Converter")
st.subheader("Smart Converter")

df = pd.read_csv("/home/gv/experiments/gbe/folder/uk_products_smart_converter.csv")
st.text("This is a text.")
st.write("Put your **text here**, **_ITALICS_**")
st.write(df)

dic = {"a": 10, "b": 20, "c": 30, "d": 40}
st.write(dic)

fig, ax = plt.subplots()
ax.scatter(np.arange(len(dic)), np.arange(len(dic))**2)
st.write(fig)

st.write("/home/gv/Downloads/ACME_MSA.pdf")

code = """def func():
    print("Hello World")"""
st.code(code, language="python")
st.text("Dataframe")
st.dataframe(df) # height and width can be adjusted

st.text("Table")
st.table(df[:5]) # no customization here


st.metric("Precision", value="92.9", delta="0.1", delta_color="normal")

data = json.load(open("/home/gv/Downloads/1697092581.json"))
st.json(data, expanded=False)

st.subheader("Line Chart")
st.line_chart(df, y=["SKU"])

st.subheader("Area Chart")
st.area_chart(df, y=["SKU"])


st.subheader("Bar Chart")
st.bar_chart(df, y=["SKU"])

st.subheader("Map")
places = pd.DataFrame({"lat": [19.07, 28.64], "lon": [72.87, 77.21]})
st.map(places)
def get_map():
    st.write(time.time())

st.write("Buttons")

button = st.button("Print Time")
if button:
    st.write(time.time())

st.subheader("Download CSV")
download = df.to_csv(index=False).encode("utf-8")
st.download_button(data=download, label="Download File", file_name="file.csv", mime="text/csv")

st.subheader("Download Txt")
txt = "this is a sample text"
st.download_button(data=txt, label="text download", file_name="file.txt")

st.subheader("Download Image")
img = open("/home/gv/Pictures/send_luka.png", "rb")
btn = st.download_button(label="Download Image", data=img, file_name="image.png", mime="image/png")


st.subheader("Check boxes")
ck = st.checkbox("Is this a checkbox?")
if ck:
    st.write("Yes")
else:
    st.write("No")

st.subheader("Radio")
radio = st.radio(label="Order your food", options=("A", "B", "C", "D"), index=1)
st.write(f"you ordered {radio}")

st.subheader("Select Box")
option = st.selectbox(label="Where do you live", options=("Moscow", "India", "USA"))

st.write(f"you live in {option}")

st.subheader("multiselect")
option = st.multiselect(label="Where do you live", options=("A", "B", "C", "D"), default=("A", "B"))
st.write(option)


st.subheader("Slider")
num = st.slider(label = "Age", min_value=0, max_value=10, step=1, value=0)
st.write(num)

st.subheader("Slider Range")
num = st.slider(label = "Slider", min_value=18, max_value=100, step=1, value=(25,35))
st.write(num)

st.subheader("Time Range")
visit_timing = st.slider(label = "Appointment",  value=(time(10,35), time(12, 50)))
st.write(visit_timing)

st.subheader("text input")
txt = st.text_input(label = "Please enter your email", max_chars=20, placeholder="email here")
st.write(txt)

st.subheader("password input")
passw = st.text_input(label = "Please enter your pass", max_chars=20, placeholder="password here", type="password")
# st.write(passw)

st.subheader("number input")

num = st.number_input(label = "Enter your age", min_value=40, max_value=120, value=60, step=2)
st.write(num)

st.subheader("Text Area")
txt = st.text_area(label="Chat here", placeholder="Chat here", height=250, max_chars=30)
st.write(txt)

import datetime
from io import StringIO
dat = st.date_input(label="Enter your birthday", value=datetime.date(2023,4,4))

dat = st.time_input(label="Enter your birthday", value=datetime.time())

st.subheader("Upload files")
fl = st.file_uploader(label="upload file",  )
if fl:
    st.write(fl.type)
    if fl.type == "text/plain":
        stringio = StringIO(fl.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)


from PIL import Image

st.subheader("Camera")
# pic = st.camera_input("Take a pic")
# if pic:
#     img = Image.open(pic)
#     st.write(img)

st.subheader("Color pic")
color_picker = st.color_picker(label="Choose")
if color_picker:
    st.write(color_picker)


#media elements
import cv2

st.subheader("Image")
img = Image.open("/home/gv/Pictures/send_luka.png")
img = cv2.imread("/home/gv/Pictures/send_luka.png") # default bgr format
st.image(img, caption="Picture", width=400, channels="BGR")


st.subheader("Audio")
# aud = st.audio("file.mp3", start_time=10, )

st.subheader("Video")
# vid = st.video("/home/gv/local/pytorch Vision/gans/output.mp4" )


# side bar
st.subheader("Sidebar")
choice = st.sidebar.radio("Choose the option", options=("audio", "video"))

# if choice == "audio":
#     st.video("/home/gv/local/pytorch Vision/gans/output.mp4")
# if choice == "video":
#         st.video("/home/gv/local/pytorch Vision/gans/output.mp4")
st.subheader("Columns")
col1, col2, col3 = st.columns(3, gap="small")
col1.video("/home/gv/local/pytorch Vision/gans/output.mp4")
col3.video("/home/gv/local/pytorch Vision/gans/output.mp4")
col2.write("/home/gv/local")

st.subheader("Tabs")
tab1, tab2 = st.tabs(["audio", "video"])
tab1.write("Text1")
tab2.video("/home/gv/local/pytorch Vision/gans/output.mp4")

st.subheader("Expander")
expan = st.expander("See pic")
expan.write("Video and Image")
expan.image("/home/gv/Pictures/send_luka.png")
expan.video("/home/gv/local/pytorch Vision/gans/output.mp4")

st.subheader("Ratio Columns")
col1, col2, col3 = st.columns([2, 2, 2], gap="small")
col1.video("/home/gv/local/pytorch Vision/gans/output.mp4")
col3.video("/home/gv/local/pytorch Vision/gans/output.mp4")
col2.write("/home/gv/local")


st.subheader("Containers")
st.write("One")
st.write("One")
st.write("One")

cont = st.container()
cont.write("1")
st.write("2")
cont.write("3")

import time
# st.subheader("status bar")
# txt = "% completed"
# mybar = st.progress(0, text=txt)
# for i in range(100):
#     time.sleep(0.1)
#     mybar.progress(i+1, text=txt)

st.subheader("Spinner")
# with st.spinner("wait for it.."):
#     time.sleep(2)
# st.write("Done!")
#
st.subheader("Balloons")
# st.balloons()
#
st.subheader("snow")
# st.snow()

st.subheader("Error")
st.error("This is an error message")

st.subheader("Warning")
st.warning("This is an error message")

st.subheader("Success")
st.success("This is an error message")

st.subheader("Info")
st.info("This is an error message")


st.subheader("RuntimeError")
e = RuntimeError("Thi is an error message")
st.exception(e)

st.subheader("Stop")
txt = st.text_area("ENter some text")
if not txt:
    st.warning("Enter text please")
    st.stop()
st.success("go ahead")

st.subheader("Form")

form = st.form("Basic Form")
name = form.text_input("Name")
age = form.slider("Age", min_value=18, max_value=60, step=1)
dob = form.date_input("DOB", value=datetime.date(2024,3,4))
submit = form.form_submit_button("Submit")
if submit:
    st.write(name, age, dob)

# st.set_page_config(page_title="New App", layout="wide")

st.subheader("Echo")

def summ(a,b):
    return a + b

with st.echo():
    su = summ(10, 17)
st.write(su)


# session states
st.subheader("Session state")
st.session_state
if "key" not in st.session_state:
    st.session_state["key"] = 1
st.session_state

if "key" in st.session_state:
    st.session_state["key"] = 2
st.session_state

# delet session state:
if "key" in st.session_state:
    del st.session_state["key"]
st.session_state

# input session
st.session_state
st.text_input("Name", key="gv")
st.session_state


# callback

def form_callback():
    st.write(st.session_state["my_slider"])
    st.write(st.session_state["my_checkbox"])

with st.form(key = "my_form"):
    slider_input = st.slider("Slider", min_value=0, max_value=100, step=1, key="my_slider")
    checkbox_input = st.checkbox("Yes or No", key="my_checkbox")
    submit_button = st.form_submit_button("Submit", on_click=form_callback)


st.subheader("Cache")
# cahce data
# cache resources for loading models etc
import time
@st.cache_resource # nothing should change even in the arguments or function
def load_model(data):
    st.write(time.time())

model = load_model(10)



@st.cache_data
def inference(data):
    st.write(time.time())
    return model





