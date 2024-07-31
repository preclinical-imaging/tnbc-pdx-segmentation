import argparse
import logging
import sys
import datetime

import requests
from requests.auth import HTTPBasicAuth

from SegmentationPipeline import SegmentationPipeline


def upload_to_xnat(host: str, username: str, password: str, project: str, session_id: str, collection_id: str,
                   dicom_seg_fp: str):
    logging.info('Uploading DICOM SEG file to XNAT')
    logging.debug(f'Host: {host}, User: {username}, Project: {project}, Session ID: {session_id}, '
                  f'DICOM SEG file: {dicom_seg_fp}')

    url = f'{host}/xapi/roi/projects/{project}/sessions/{session_id}/collections/{collection_id}'
    params = {
        'overwrite': 'true',
        'type': 'SEG'
    }
    auth = HTTPBasicAuth(username, password)
    headers = {
        "Content-Type": "application/octet-stream"
    }

    with open(dicom_seg_fp, "rb") as file:
        response = requests.put(url, params=params, data=file, headers=headers, auth=auth)

        if response.ok:
            logging.info('DICOM SEG file uploaded successfully')
        else:
            logging.error(f'Error uploading DICOM SEG file: {response.text}')
            response.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description='Run the TNBC segmentation pipeline then create an XNAT ROI Collection'
                                                 ' assessor XML file from the resulting DICOM SEG file')
    parser.add_argument('-s', '--host', required=True, help='XNAT host URL')
    parser.add_argument('-u', '--username', required=True, help='XNAT username')
    parser.add_argument('-p', '--password', required=True, help='XNAT password')
    parser.add_argument('-r', '--project', required=True, help='XNAT project ID')
    parser.add_argument('-i', '--session_id', required=True, help='XNAT session ID')
    parser.add_argument('-d', '--input_dir', required=True, help='Input directory containing DICOM files')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for DICOM SEG file')
    parser.add_argument('-w', '--weight_dir', required=False, default='./weights', help='Path to weights directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    kwargs = vars(parser.parse_args())

    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if kwargs['verbose']:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        pipeline = SegmentationPipeline(**kwargs)
        pipeline.run_pipeline()

        dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        for (series, segmentation_tup) in pipeline.dicom_segmentations.items():
            series_short = series.split('.')[-1]
            for (fp, seg_ds) in segmentation_tup:
                upload_to_xnat(kwargs['host'], kwargs['username'], kwargs['password'], kwargs['project'],
                               kwargs['session_id'], f'DR2UnetSegmentation_{series_short}_{dt}', fp)

    except Exception as e:
        logging.error('Error running pipeline: %s', e)
        logging.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()