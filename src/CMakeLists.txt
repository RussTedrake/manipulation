# -*- mode: cmake -*-
# vi: set ft=cmake :

# Copyright (c) 2019, Massachusetts Institute of Technology.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

function(add_python_test FILENAME)
  get_filename_component(CURRENT_DIRECTORY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}"
    NAME
  )
  set(TEST_NAME "${CURRENT_DIRECTORY_NAME}_${FILENAME}")
  set(Python_ADDITIONAL_VERSIONS 3.8)
  find_package(PythonInterp 3.6 MODULE REQUIRED)
  add_test(NAME "${TEST_NAME}"
    COMMAND "${PYTHON_EXECUTABLE}"
      -b
      "${FILENAME}"
      ${ARGN}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  find_package(drake CONFIG REQUIRED)
  get_filename_component(DRAKE_PYTHONPATH
    "${drake_DIR}/../../python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages"
    REALPATH
  )
  set(PYTHONPATH
    "${PROJECT_SOURCE_DIR}/src:${PROJECT_SOURCE_DIR}/underactuated/src:${DRAKE_PYTHONPATH}:$ENV{PYTHONPATH}"
  )
  set_tests_properties("${TEST_NAME}" PROPERTIES
    ENVIRONMENT "MPLBACKEND=Template;PYTHONPATH=${PYTHONPATH}"
    FAIL_REGULAR_EXPRESSION DrakeDeprecationWarning
    LABELS unit
    TIMEOUT 60
  )
endfunction()

function(add_python_style_test)
  get_filename_component(CURRENT_DIRECTORY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}"
    NAME
  )
  set(TEST_NAME "${CURRENT_DIRECTORY_NAME}_style")
  find_program(PYCODESTYLE_EXECUTABLE NAMES pycodestyle pep8)
  if(NOT PYCODESTYLE_EXECUTABLE)
    message(FATAL_ERROR "pycodestyle was NOT found")
  endif()
  add_test(NAME "${TEST_NAME}"
    COMMAND "${PYCODESTYLE_EXECUTABLE}"
      --count
      --show-source
      "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties("${TEST_NAME}" PROPERTIES
    LABELS style
    TIMEOUT 60
  )
endfunction()

add_subdirectory(depth_camera_demo)
add_subdirectory(planar_iiwa)
