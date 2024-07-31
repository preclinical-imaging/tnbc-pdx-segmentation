FROM xnat/tensorflow-notebook:develop

USER root

# Update the base image
RUN apt-get update &&  \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install the required packages
RUN python3 -m pip install lxml && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER root

# Set the working directory and copy requirements
WORKDIR /usr/src/app

# Copy the model weights and source code
RUN mkdir -p /usr/src/app/weights
COPY weights_new/ ./weights/
COPY *.py ./
COPY *.npy ./
COPY *.mat ./

RUN chown -R ${NB_UID}:${NB_GID} /usr/src/app

ENV PYTHONPATH=${PYTHONPATH}:/usr/src/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/src/app:$PATH

USER ${NB_UID}

# Set the entrypoint and command
ENTRYPOINT [ "python", "main_xnat.py" ]
CMD ["--help"]

# Label is a JSON string for XNAT's container service to parse
LABEL org.nrg.commands="[{\"name\": \"dr2unet-for-tnbc-pdx-segmentation\", \"label\": \"dr2unet-for-tnbc-pdx-segmentation\", \"description\": \"Segment a TNBC PDX image session using DR2UNET model\", \"version\": \"0.0.1\", \"schema-version\": \"1.0\", \"info-url\": \"https://github.com/preclinical-imaging/DR2UNet-for-TNBC-PDX-Tumor-Segmentation.git\", \"container-name\": \"\", \"type\": \"docker\", \"index\": \"\", \"working-directory\": \"/usr/src/app\", \"command-line\": \"python run.py /input/SCANS /output --log-level DEBUG -u \$XNAT_USER -p \$XNAT_PASS -s \$XNAT_HOST #PROJECT_ID# #SESSION_LABEL#\", \"override-entrypoint\": true, \"mounts\": [{\"name\": \"input-mount\", \"writable\": false, \"path\": \"/input\"}, {\"name\": \"output-mount\", \"writable\": true, \"path\": \"/output\"}], \"environment-variables\": {}, \"ports\": {}, \"inputs\": [{\"name\": \"PROJ\", \"description\": \"Project\", \"type\": \"string\", \"replacement-key\": \"#PROJ#\", \"command-line-flag\": \"-p\", \"required\": true}, {\"name\": \"SUBJ_ID\", \"description\": \"Subject ID\", \"type\": \"string\", \"required\": true}, {\"name\": \"SESS_ID\", \"description\": \"Session ID\", \"type\": \"string\", \"required\": true}, {\"name\": \"SESS_LABEL\", \"description\": \"Session Label\", \"type\": \"string\", \"required\": true}], \"outputs\": [{\"name\": \"xml\", \"description\": \"The assessor XML\", \"mount\": \"out\", \"path\": \"assessors/\", \"required\": true}, {\"name\": \"dicom-segmentations\", \"description\": \"The DICOM segmentation files\", \"mount\": \"out\", \"required\": true, \"path\": \"dicom/\"}], \"xnat\": [{\"name\": \"dr2unet-for-tnbc-pdx-segmentation\", \"label\": \"DR2UNET for TNBC PDX Segmentation\", \"description\": \"Segment a TNBC PDX image session using DR2UNET model\", \"contexts\": [\"xnat:mrSessionData\"], \"external-inputs\": [{\"name\": \"session\", \"label\": \"Session\", \"description\": \"Session\", \"type\": \"Session\", \"required\": true, \"provides-files-for-command-mount\": \"input-mount\", \"load-children\": true}], \"derived-inputs\": [{\"name\": \"project\", \"label\": \"Project\", \"description\": \"Project\", \"type\": \"string\", \"required\": true, \"user-settable\": false, \"derived-from-wrapper-input\": \"session\", \"derived-from-xnat-object-property\": \"project-id\", \"provides-value-for-command-input\": \"PROJ\"}, {\"name\": \"subject-id\", \"label\": \"Subject ID\", \"description\": \"Subject ID\", \"type\": \"string\", \"required\": true, \"user-settable\": false, \"derived-from-wrapper-input\": \"session\", \"derived-from-xnat-object-property\": \"subject-id\", \"provides-value-for-command-input\": \"SUBJ_ID\"}, {\"name\": \"session-id\", \"label\": \"Session ID\", \"description\": \"Session ID\", \"type\": \"string\", \"required\": true, \"user-settable\": false, \"derived-from-wrapper-input\": \"session\", \"derived-from-xnat-object-property\": \"id\", \"provides-value-for-command-input\": \"SESS_ID\"}, {\"name\": \"session-label\", \"label\": \"Session Label\", \"description\": \"Session Label\", \"type\": \"string\", \"required\": true, \"user-settable\": false, \"provides-value-for-command-input\": \"SESS_LABEL\", \"derived-from-wrapper-input\": \"session\", \"derived-from-xnat-object-property\": \"label\"}], \"output-handlers\": [{\"name\": \"assessor\", \"type\": \"Assessor\", \"accepts-command-output\": \"xml\", \"as-a-child-of\": \"session\"}, {\"name\": \"assessor-resource\", \"type\": \"Resource\", \"accepts-command-output\": \"dicom-segmentations\", \"as-a-child-of\": \"assessor\", \"label\": \"SEG\"}]}]}]"
