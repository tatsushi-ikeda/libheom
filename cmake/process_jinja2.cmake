# -*- mode: cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

set(J2_SCRIPT "${PROJECT_SOURCE_DIR}/tools/process_jinja2.py" CACHE INTERNAL "" FORCE)
set(J2_PARAMS "${PROJECT_BINARY_DIR}/tools/params.json" CACHE INTERNAL "" FORCE)

function(process_jinja2 filename)
  cmake_parse_arguments(J2 "" "SOURCE" "" ${ARGN})
  if(NOT J2_SOURCE)
    set(J2_SOURCE "${filename}.j2")
  endif()
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
    COMMAND
    "${PYTHON_EXECUTABLE}"
    "${J2_SCRIPT}"
    "${CMAKE_CURRENT_LIST_DIR}/${J2_SOURCE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
    "${J2_PARAMS}"
    DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/${J2_SOURCE}"
    "${J2_PARAMS}")
endfunction()
