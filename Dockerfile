FROM python:3.10.14-slim-bookworm
RUN mkdir /service
WORKDIR /service
COPY ./requirements.txt ./
#RUN pip3 install --no-cache-dir --index-url=http://pypi-proxy:5001/index/ --trusted-host=pypi-proxy --use-deprecated=legacy-resolver -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
#Till here, keep build steps that do not change frequently.
#From here, Copy the codebase/files that change frequently.
COPY . .
ENTRYPOINT flower-superlink --insecure
