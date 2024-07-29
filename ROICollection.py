import argparse
import os
import pydicom
import datetime as dt
import sys
import uuid

from lxml.builder import ElementMaker
from lxml.etree import tostring as xmltostring

import logging


class ROICollection:

    def __init__(self, project, subject_id, session_id, session_label, dicom_seg_in, assessor_xml_out, **kwargs):
        self.project = project
        self.subject_id = subject_id
        self.session_id = session_id
        self.session_label = session_label
        self.dicom_seg_in = dicom_seg_in
        self.assessor_xml_out = assessor_xml_out
        self.assessor_xml = None

        logging.debug('Creating ROICollection with project=%s, subject_id=%s, session_id=%s, '
                      'session_label=%s, dicom_seg_in=%s, assessor_xml_out=%s',
                      project, subject_id, session_id, session_label, dicom_seg_in, assessor_xml_out)

        logging.debug('Reading DICOM SEG file %s', dicom_seg_in)

        self.dicom_seg_ds = pydicom.dcmread(dicom_seg_in)

        logging.debug('DICOM SEG file read successfully')

    def build(self):
        logging.info('Building assessor XML')

        nsdict = {'xnat': 'http://nrg.wustl.edu/xnat',
                  'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                  'icr': 'http://icr.ac.uk/icr'}

        xnat_host = os.environ.get('XNAT_HOST', 'http://nrg.wustl.edu')
        schema_location_template = "{0} {1}/xapi/schemas/{2}/{2}.xsd"
        schema_location = schema_location_template.format(nsdict['xnat'], xnat_host, 'xnat') + ' ' + \
                          schema_location_template.format(nsdict['icr'], xnat_host, 'roi')

        def ns(namespace, tag):
            return "{%s}%s" % (nsdict[namespace], tag)

        uid = self.dicom_seg_ds.SOPInstanceUID
        reference_uid = self.dicom_seg_ds.ReferencedSeriesSequence[0].SeriesInstanceUID
        name = getattr(self.dicom_seg_ds, 'SeriesDescription', 'Segmentation')
        date = getattr(self.dicom_seg_ds, 'StudyDate', dt.datetime.now().strftime('%Y%m%d'))
        time = getattr(self.dicom_seg_ds, 'StudyTime', dt.datetime.now().strftime('%H%M%S'))

        assessor_label = f"SEG_{date}_{time}_{uuid.uuid4().hex[:3]}_S{reference_uid.split('.')[-1]}"
        assessor_id = f"RoiCollection_{uuid.uuid4().hex[:6]}_{uuid.uuid4().hex[:10]}"

        assessorElements = [
            ('UID', uid),
            ('collectionType', 'SEG'),
            ('subjectID', self.subject_id),
            ('name', name)
        ]

        logging.debug('Building assessor XML with elements %s', assessorElements)

        assessorTitleAttributesDict = {
            'ID': assessor_id,
            'label': assessor_label,
            'project': self.project,
            ns('xsi', 'schemaLocation'): schema_location
        }

        E = ElementMaker(namespace=nsdict['icr'], nsmap=nsdict)
        assessorXML = E('RoiCollection', assessorTitleAttributesDict,
                        E(ns('xnat', 'date'), dt.date.today().isoformat()),
                        E(ns('xnat', 'time'), dt.datetime.now().strftime('%H:%M:%S')),
                        E(ns('xnat', 'imageSession_ID'), self.session_id),
                        E('UID', uid),
                        E('collectionType', 'SEG'),
                        E('subjectID', self.subject_id),
                        E('references', *[E('seriesUID', reference_uid)]),
                        E('name', name)
                        )

        logging.debug('Assessor XML built successfully')

        self.assessor_xml = assessorXML

    def to_string(self):
        return xmltostring(self.assessor_xml, pretty_print=True, encoding='UTF-8', xml_declaration=True).decode()

    def write(self):
        logging.info(f'Writing assessor XML to {self.assessor_xml_out}')

        if self.assessor_xml is not None:
            with open(self.assessor_xml_out, 'w') as f:
                f.write(self.to_string())
            logging.debug('Assessor XML written successfully')
        else:
            logging.error('Assessor XML not built, cannot write to file')
            raise ValueError('Assessor XML not built, cannot write to file')

    def run(self):
        self.build()
        self.write()

    @staticmethod
    def create_assessor(project, subject_id, session_id, session_label, dicom_seg_in, assessor_xml_out, **kwargs):
        assessor = ROICollection(project, subject_id, session_id, session_label,
                                 dicom_seg_in, assessor_xml_out, **kwargs)
        assessor.run()
        return assessor


def main():
    parser = argparse.ArgumentParser(description='Create an XNAT ROI Collection assessor XML file from a DICOM SEG file')
    parser.add_argument('-p', '--project', required=True, help='XNAT project ID')
    parser.add_argument('-s', '--subject_id', required=True, help='XNAT subject ID')
    parser.add_argument('-i', '--session_id', required=True, help='XNAT session ID')
    parser.add_argument('-l', '--session_label', required=True, help='XNAT session label')
    parser.add_argument('-d', '--dicom_seg_in', required=True, help='Input DICOM SEG file')
    parser.add_argument('-o', '--assessor_xml_out', required=True, help='Output XNAT assessor XML file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    kwargs = vars(parser.parse_args())



    if kwargs['verbose']:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        ROICollection.create_assessor(**kwargs)
    except Exception as e:
        logging.error('Error creating assessor: %s', e)
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
