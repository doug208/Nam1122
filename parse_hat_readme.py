
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;


#!/usr/bin/env python3
"""Parse HAT README for model download links"""

import urllib.request
import re

url = "https://raw.githubusercontent.com/chxy95/HAT/master/README.md"
try:
    with urllib.request.urlopen(url, timeout=15) as response:
        content = response.read().decode()
        
        # Print sections about models
        if 'Model Zoo' in content:
            start = content.find('## Model Zoo')
            if start == -1:
                start = content.lower().find('## model zoo')
            if start != -1:
                # Find the next section or end of file
                next_section = content.find('## ', start+1)
                if next_section == -1:
                    next_section = len(content)
                section = content[start:next_section]
                print("=== Model Zoo Section ===")
                print(section)
                
        # Look for any table or list with model names and links
        print("\n\n=== Looking for Model Links ===")
        
        # Find markdown links with .pth
        pth_links = re.findall(r'\[([^\]]+)\]\((https?://[^)]+\.pth)\)', content)
        print(f"\nMarkdown .pth links found: {len(pth_links)}")
        for name, link in pth_links[:10]:
            print(f"  - {name}: {link}")
            
        # Find raw URLs with .pth
        raw_pth = re.findall(r'https?://[^\s\"\'<>]+\.pth', content)
        print(f"\nRaw .pth URLs found: {len(raw_pth)}")
        for link in set(raw_pth):
            if 'hat' in link.lower() or 'HAT' in link:
                print(f"  - {link}")
                
        # Find Google Drive file IDs
        gdrive_files = re.findall(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', content)
        gdrive_folders = re.findall(r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)', content)
        print(f"\nGoogle Drive file IDs: {gdrive_files}")
        print(f"Google Drive folder IDs: {gdrive_folders}")
        
        # Try to get direct download links
        for file_id in gdrive_files[:3]:
            print(f"\n  Potential direct link for {file_id}:")
            print(f"  https://drive.google.com/uc?export=download&id={file_id}")
            
except Exception as e:
    print(f"Error: {e}")