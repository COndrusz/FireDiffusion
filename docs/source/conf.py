import sys
import os

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

author = 'Christopher Ondrusz'
project = 'Fire Diff'
version = '1.0'
release = '1.0.0'
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
templates_path = ['_templates']
html_css_files = [
    'custom.css',
]
html_theme = 'pyramid'
autodoc_default_options = {
    'exclude-members': '__weakref__, note',
}
latex_elements = {
    'papersize': 'a4paper',  # or 'letterpaper'
    'pointsize': '10pt',
    'preamble': r'''
% Custom LaTeX preamble if needed
''',
}
