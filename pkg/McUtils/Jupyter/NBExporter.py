from nbconvert.preprocessors.extractoutput import *

class MarkdownImageExtractor(ExtractOutputPreprocessor):
    """
    Extracts all of the outputs from the notebook file.  The extracted
    outputs are returned in the 'resources' dictionary.
    """

    def __init__(self, *args, prefix='', **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    def reencode_data(self, mime_type, data):
        # Binary files are base64-encoded, SVG is already XML
        if mime_type in {'image/png', 'image/jpeg', 'application/pdf'}:
            # data is b64-encoded as text (str, unicode),
            # we want the original bytes
            data = a2b_base64(data)
        elif mime_type == 'application/json' or not isinstance(data, text_type):
            # Data is either JSON-like and was parsed into a Python
            # object according to the spec, or data is for sure
            # JSON. In the latter case we want to go extra sure that
            # we enclose a scalar string value into extra quotes by
            # serializing it properly.
            if isinstance(data, bytes) and not isinstance(data, text_type):
                # In python 3 we need to guess the encoding in this
                # instance. Some modules that return raw data like
                # svg can leave the data in byte form instead of str
                data = data.decode('utf-8')
            data = platform_utf_8_encode(json.dumps(data))
        else:
            # All other text_type data will fall into this path
            data = platform_utf_8_encode(data)
        return data

    def save_attachement(self, filename, attachement, index, cell, resources, cell_index):

        # Get the unique key from the resource dict if it exists.  If it does not
        # exist, use 'output' as the default.  Also, get files directory if it
        # has been specified
        unique_key = resources.get('unique_key', 'output')
        output_files_dir = resources.get('output_files_dir', None)

        mime_type = next(iter(attachement.keys()))
        data = next(iter(attachement.values()))

        # ext = guess_extension_without_jpe(mime_type)
        # if ext is None:
        #     ext = '.' + mime_type.rsplit('/')[-1]
        # if out.metadata.get('filename', ''):
        #     filename = out.metadata['filename']
        #     if not filename.endswith(ext):
        #         filename += ext
        # else:
        # filename = self.output_filename_template.format(
        #     unique_key=unique_key,
        #     cell_index=cell_index,
        #     index=index,
        #     extension=ext)

        resources['outputs'][filename] = self.reencode_data(mime_type, data)
        return filename

    def preprocess_cell(self, cell, resources, cell_index):
        if cell['cell_type'] == 'markdown':
            if 'attachments' in cell:
                for index, (name, attachement) in enumerate(cell['attachments'].items()):
                    fn = self.save_attachement(name, attachement, index, cell, resources, cell_index)
                    cell['source'] = cell['source'].replace(
                        'attachment:'+name,
                        self.prefix+fn
                    )
        elif cell['cell_type'] == 'code' and cell['source'].strip() == "":
            cell['cell_type'] = 'raw'
            cell['source'] = cell['source'].strip()
        else:
            cell, resources = super().preprocess_cell(cell, resources, cell_index)

        return cell, resources
