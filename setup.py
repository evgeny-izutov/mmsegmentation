# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os
import os.path as osp
from setuptools import find_packages, setup


try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def readme():
    root_dir_path = osp.dirname(osp.realpath(__file__))
    readme_file_path = osp.join(root_dir_path, 'README.md')

    with open(readme_file_path, encoding='utf-8') as f:
        content = f.read()

    return content


def get_version():
    root_dir_path = osp.dirname(osp.realpath(__file__))
    version_file_path = osp.join(root_dir_path, 'mmseg/version.py')

    with open(version_file_path, 'r') as f:
        exec(compile(f.read(), version_file_path, 'exec'))

    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re

    root_dir_path = osp.dirname(osp.realpath(__file__))
    require_fpath = osp.join(root_dir_path, fname)

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())

    return packages


def get_extensions():
    extensions = []

    ext_name = 'mmseg._mpl'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    extra_compile_args = {'cxx': []}

    print(f'Compiling {ext_name} without CUDA')
    op_files = glob.glob('./mmseg/ops/csrc/*.cpp')
    include_path = os.path.abspath('./mmseg/ops/csrc')
    ext_ops = CppExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=[],
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    setup(
        name='otxseg',
        version='0.3.0',
        description='Temporary OTX fork of MMSegmenation Semantic Segmentation Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab, Intel',
        author_email='songki.choi@intel.com',
        keywords='computer vision, semantic segmentation',
        url='http://github.com/openvinotoolkit/mmsegmentation',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=get_extensions(),
        cmdclass=cmd_class,
        zip_safe=False)
