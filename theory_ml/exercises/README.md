exercises.pdf presents theory and problem statements

Ch*.ipynb files are the files taken from the course exercises. \
Link to the book as well as these exercises: \
(link to book) \
(link to exercises) \




Step 0: install uv: \
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" \
(close the current cmd shell or powershell and open it again.) \

uv python install 3.12.12 \

Step 1: setup a virtual environment: \
uv venv \
.venv\Scripts\activate \

uv pip install islp \ 
uv pip install pandas \
uv pip install ipykernel \
etc.... \



# Alternative install without uv \
Download and install python: (link)\
Create a virtual environment. \
Activate it. \
Run "pip install -e . --config-settings editable_mode=compat" \




# Useful links: \
https://islp.readthedocs.io/en/latest/installation.html \
