#!/bin/sh
# shellcheck shell=sh
# shellcheck source=/dev/null
# shellcheck disable=SC2312

# Copyright 2019 Intel Corporation
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ############################################################################

# Overview

# This is the top-level environment variable setup script for use with Intel
# oneAPI toolkits. Most "tools" or "components" that are part of an Intel
# oneAPI installation include an `env/vars.sh` script that configures the
# environment variables needed for that specific tool to work. More details
# are available via this script's "--help" argument.

# NOTE: This script is designed to be POSIX compatible, so that it can more
# easily be _sourced_ by a variety of shell scripts (zsh, ksh, bash, etc.) on
# a variety of UNIX-style systems (macOS and Linux). The "shebang" on line one
# of this script has no bearing on the actual shell that is used to execute
# this script, because it will always be executed within the context of the
# shell from which it is sourced.


# ############################################################################

# Developed on Ubuntu 18.04.5 LTS with the following shells:
#   bash: 4.4.20(1)-release
#   dash: 0.5.8-2.10
#    zsh: 5.4.2
#    ksh: Version AJM 93u+ 2012-08-01
#   lksh: @(#)LEGACY KSH R56 2018/01/14
#   mksh: @(#)MIRBSD KSH R56 2018/01/14
#
# Developed on macOS version 10.15.7 build 19H15 with the following shells:
#
#   bash: 3.2.57(1)-release
#   bash: 5.0.18(1)-release
#   dash: unknown version
#   zsh: 5.7.1
#   ksh: Version AJM 93u+ 2012-08-01


# ############################################################################

# Globals

# Name of this script is needed to reliably detect source with /bin/sh.
# Knowing the name of this script is a reasonable assumption because several
# of our extension/plugins assume this well-known name for referencing this
# script. This script is designed to be compatible with /bin/sh so it works
# across multiple Linux/BSD shells, in particular: bash and zsh.

script_name=setvars.sh
config_file=""
config_array=""
component_array=""
warning_tally=0
posix_nl='
'

# TODO: create an enumerated list of script return codes.


# ############################################################################

# To be called if we encounter bad command-line args or user asks for help.

# Inputs:
#   none
#
# Outputs:
#   message to stdout

usage() {
  echo "  "
  echo "usage: source ${script_name}" '[--force] [--config=file] [--help] [...]'
  echo "  --force        Force ${script_name} to re-run, doing so may overload environment."
  echo "  --config=file  Customize env vars using a ${script_name} configuration file."
  echo "  --help         Display this help message and exit."
  echo "  ...            Additional args are passed to individual env/vars.sh scripts"
  echo "                 and should follow this script's arguments."
  echo "  "
  echo "  Some POSIX shells do not accept command-line options. In that case, you can pass"
  echo "  command-line options via the SETVARS_ARGS environment variable. For example:"
  echo "  "
  echo "  $ SETVARS_ARGS=\"ia32 --config=config.txt\" ; export SETVARS_ARGS"
  echo "  $ . path/to/${script_name}"
  echo "  "
  echo "  The SETVARS_ARGS environment variable is cleared on exiting ${script_name}."
  echo "  "
}


# ############################################################################

# To be called in preparation to exit this script, on error or success.

# Usage:
#   # Restore original $@ array before return.
#   eval set -- "$script_args" || true
#   prep_for_exit <return-code> ; return
#
# Without a "; return" immediately following the call to this function, this
# sourced script will not exit!! Using the "exit" command causes a sourced
# script to exit the containing terminal session.
#
# IMPORTANT: The 'eval set ...' statement must be placed as above, directly
# above the prep_for_exit() call. This will restore the original caller's
# $@ array (their original argument list). This cannot be done inside of the
# prep_for_exit() function because it restores to a private copy of the $@
# array, which is lost upon the return from the function, so it must be
# placed inline, as shown above. :-(
#
# Inputs:
#   Expects $1 to specify a return code. A "0" is considered a success.
#
# Outputs:
#   return code (provided as input or augmented when none was provided)

# For POSIX, limit utility usage to "native" core utilities in this script.
# Especially important for macOS where Brew may have replaced core utilities.

prep_for_exit() {
  script_return_code=$1

  unset -v SETVARS_CALL || true
  unset -v SETVARS_ARGS || true
  unset -v SETVARS_VARS_PATH || true

  # make sure we're dealing with numbers
  # TODO: add check for non-numeric return codes.
  if [ "$script_return_code" = "" ] ; then
    script_return_code=255
  fi

  if [ "$script_return_code" -eq 0 ] ; then
    SETVARS_COMPLETED=1 ; export SETVARS_COMPLETED
  fi

  return "$script_return_code"
}


# ############################################################################

# Since zsh manages the expansion of for loop expressions differently than
# bash, ksh and sh, we must use the "for arg do" loop (no "in" operator) that
# implicitly relies on the positional arguments array ($@). There is only one
# $@ array; this function saves that array in a format that can be easily
# restored to the $@ array at a later time.

# see http://www.etalabs.net/sh_tricks.html ("Working with arrays" section)

# Usage:
#   array_var=$(save_args "$@")
#   eval "set -- $array_var" # restores array to the $@ variable
#
# Inputs:
#   The $@ array.
#
# Outputs:
#   Cleverly encoded string that represents the $@ array.

save_args() {
  for arg do
    printf "%s\n" "$arg" | sed -e "s/'/'\\\\''/g" -e "1s/^/'/" -e "\$s/\$/' \\\\/" ;
  done
  # echo needed to pickup final continuation "\" so it's not added as an arg
  echo " "
}

# Save a copy of the arguments array ($@) passed to this script so we can
# restore it, if needed later.
script_args=$(save_args "$@")


# ############################################################################

# Convert a list of '\n' terminated strings into a format that can be moved
# into the positional arguments array ($@) using the eval "set -- $array_var"
# command. It removes blank lines from the list (awk 'NF') in the process. It
# is not possible to combine the prep and eval steps into a single function
# because you lose the context that contains the resulting "$@" array upon
# return from this function.

# Usage:
#   eval set -- "$(prep_for_eval "$list_of_strings_with_nl")"
#
# Inputs:
#   The passed parameter is expected to be a collection of '\n' terminated
#   strings (e.g., such as from a find, ls or grep command).
#
# Outputs:
#   Cleverly encoded string that represents the $@ array.

prep_for_eval() {
  echo "$1" | awk 'NF' | sed -e "s/^/'/g" -e "s/$/' \\\/g" -e '$s/\\$//'
}


# ############################################################################

# Get absolute path to this script.
# Uses `readlink` to remove links and `pwd -P` to turn into an absolute path.

# Usage:
#   script_dir=$(get_script_path "$script_rel_path")
#
# Inputs:
#   script/relative/pathname/scriptname
#
# Outputs:
#   /script/absolute/pathname

# executing function in a *subshell* to localize vars and effects on `cd`

# The sequence `builtin cd` needs to be used with zsh instead of `command cd`
# to remove alias/function redefinitions of the `cd` command. This is because
# `command` only works with external commands in the zsh shell. Unfortunately,
# `builtin` is not recognized by `dash` and fails. Thus it is necessary to
# create two branches in the function to insure that an existing redefinition
# of `cd` (alias or function) does not interfere.

get_script_path() (
  script="$1"
  while [ -L "$script" ] ; do
    script_dir=$(command dirname -- "$script")
    # TODO: coordinate with component env/vars.sh scripts to fix throughout
    # see: https://superuser.com/a/1574553/229501
    if [ -n "${ZSH_VERSION:-}" ] ; then
      script_dir=$(builtin cd "$script_dir" && command pwd -P)
    else
      script_dir=$(command cd "$script_dir" && command pwd -P)
    fi
    script="$(readlink "$script")"
    case $script in
      (/*) ;;
       (*) script="$script_dir/$script" ;;
    esac
  done
  script_dir=$(command dirname -- "$script")
  # TODO: coordinate with component env/vars.sh scripts to fix throughout
  # see: https://superuser.com/a/1574553/229501
  if [ -n "${ZSH_VERSION:-}" ] ; then
    script_dir=$(builtin cd "$script_dir" && command pwd -P)
  else
    script_dir=$(command cd "$script_dir" && command pwd -P)
  fi
  printf "%s" "$script_dir"
)


# ############################################################################

# Determine if we are being executed or sourced. Need to detect being sourced
# within an executed script, which can happen on a CI system. We also must
# detect being sourced at a shell prompt (CLI).

# We are assuming we know the name of this script, which is a reasonable
# assumption. This script is expected to be named "setvars.sh". Making this
# assumption simplifies the process of detecting if the script has been
# sourced or executed. It also simplifies the process of detecting the
# location of this script.

# Using `readlink` to remove possible symlinks in the name of the script.
# Also, "ps -o comm=" is limited to a 15 character result, but it works
# fine here, because we are only looking for the name of this script or the
# name of the execution shell, both always fit into fifteen characters.

# TODO: Edge cases exist when executed by way of "/bin/sh setvars.sh"
# Most shells detect or fall thru to error message, sometimes ksh does not.
# This is an odd and unusual situation; not a high priority TODO.

_setvars_get_proc_name() {
  if [ -n "${ZSH_VERSION:-}" ] ; then
    script="$(ps -p "$$" -o comm=)"
  else
    script="$1"
    while [ -L "$script" ] ; do
      script="$(readlink "$script")"
    done
  fi
  basename -- "$script"
}

_setvars_this_script_name="setvars.sh"
if [ "$_setvars_this_script_name" = "$(_setvars_get_proc_name "$0")" ] ; then
  echo " "
  echo ":: ERROR: Incorrect usage: this script must be sourced."
  usage
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 255 ; exit
fi


# ############################################################################

# Extract the name and location of this sourced script.

# We've already determined that this script is being sourced, which impacts
# some of the logic being applied in the following code.

# Generally, "ps -o comm=" is limited to a 15 character result, but it works
# fine for this usage, because we are primarily interested in finding the name
# of the execution shell, not the name of any calling script.

sourcer="" ;
sourced_nm=""
sourced_sh="$(ps -p "$$" -o comm=)" ;
proc_name="$(_setvars_get_proc_name "$0")"

# ${var:-} needed to pass "set -eu" checks
# see https://unix.stackexchange.com/a/381465/103967
# see https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02
if [ -n "${ZSH_VERSION:-}" ] && [ -n "${ZSH_EVAL_CONTEXT:-}" ] ; then     # zsh 5.x and later
  sourcer=$(printf "%s: %s" "${proc_name}" "ZSH_VERSION = ${ZSH_VERSION}") ;
  # shellcheck disable=SC2249,SC2296
  case $ZSH_EVAL_CONTEXT in (*:file*) sourced_nm="${(%):-%x}" ;; esac ;
elif [ -n "${KSH_VERSION:-}" ] ; then                                     # ksh or mksh or lksh
  sourcer=$(printf "%s: %s" "${proc_name}" "KSH_VERSION = ${KSH_VERSION}") ;
  if [ "$(set | grep -Fq "KSH_VERSION=.sh.version" ; echo $?)" -eq 0 ] ; then # ksh
    # shellcheck disable=SC2296
    sourced_nm="${.sh.file}" ;
  else # mksh or lksh or [lm]ksh masquerading as ksh or sh
    # shellcheck disable=SC2296
    # force [lm]ksh to issue error msg; which contains this script's rel/path/filename, e.g.:
    # mksh: /home/ubuntu/intel/oneapi/setvars.sh[250]: ${.sh.file}: bad substitution
    sourced_nm="$( (echo "${.sh.file}") 2>&1 )" || : ;
    sourced_nm="$(expr "${sourced_nm:-}" : '^.*sh: \(.*\)\[[0-9]*\]:')" ;
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then      # bash
  sourcer=$(printf "%s: %s" "${proc_name}" "BASH_VERSION = ${BASH_VERSION}") ;
  # shellcheck disable=SC2128,SC3028
  (return 0 2>/dev/null) && sourced_nm="${BASH_SOURCE}" ;
elif [ "dash" = "${sourced_sh:-}" ] ; then  # dash
  sourcer=$(printf "%s: %s" "${proc_name}" "DASH_VERSION = unknown") ;
  # shellcheck disable=SC2296
  # force dash to issue error msg; which contains this script's rel/path/filename, e.g.:
  # dash: 266: /home/ubuntu/intel/oneapi/setvars.sh: Bad substitution
  sourced_nm="$( (echo "${.sh.file}") 2>&1 )" || : ;
  sourced_nm="$(expr "${sourced_nm:-}" : '^.*dash: [0-9]*: \(.*\):')" ;
elif [ "sh" = "${sourced_sh:-}" ] ; then    # could be dash masquerading as /bin/sh
  sourcer=$(printf "%s: %s" "${proc_name}" "SH_VERSION = unknown") ;
  # shellcheck disable=SC2296
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "setvars.sh"
  sourced_nm="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$sourced_nm" | grep -Eq "sh: [0-9]+: .*setvars\.sh: " ; echo $?)" -eq 0 ] ; then # dash as sh
    # sh: 155: /home/ubuntu/intel/oneapi/setvars.sh: Bad substitution
    sourced_nm="$(expr "${sourced_nm:-}" : '^.*sh: [0-9]*: \(.*\):')" ;
  fi
else  # unrecognized shell or dash being sourced from within a user's script
  sourcer=$(printf "%s: %s" "${proc_name}" "???_VERSION = unknown") ;
  # shellcheck disable=SC2296
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "setvars.sh"
  sourced_nm="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$sourced_nm" | grep -Eq "^.+: [0-9]+: .*setvars\.sh: " ; echo $?)" -eq 0 ] ; then # dash
    # .*: 164: intel/oneapi/vars.sh: Bad substitution
    sourced_nm="$(expr "${sourced_nm:-}" : '^.*: [0-9]*: \(.*\):')" ;
  else
    sourced_nm="" ;
  fi
fi

if [ "" = "$sourced_nm" ] ; then
  echo " "
  >&2 echo ":: ERROR: Unable to proceed: possible causes listed below."
  >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
  >&2 echo "   Unrecognized/unsupported shell (supported: bash, zsh, ksh, m/lksh, dash)." ;
  >&2 echo "   May fail in dash if you rename this script (assumes \"setvars.sh\")." ;
  >&2 echo "   Can be caused by sourcing from ZSH version 4.x or older." ;
  usage
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 255 ; return
fi

# Determine path to this file ($script_name).
# Expects to be located at the top (root) of the oneAPI install directory.
script_root=$(get_script_path "${sourced_nm:-}")


# ############################################################################

# Interpret command-line arguments passed to this script and remove them.
# Ignore unrecognized CLI args, they will be passed to the env/vars scripts.
# see https://unix.stackexchange.com/a/258514/103967

help=0
force=0
config=0
config_file=""
list_of_args=""

if [ -n "${SETVARS_ARGS:-}" ] ; then              # use arguments found in $SETVARS_ARGS
  input_args=$(printf "%s" "args: Using \"SETVARS_ARGS\" for ${script_name} arguments: $SETVARS_ARGS")
  list_of_args=$(echo "$SETVARS_ARGS" | tr -s "[:blank:]" \\n | awk 'NF') # convert args to \n terminated lines
  eval set -- "$(prep_for_eval "$list_of_args")"  # copy that list into the $@ array
else
  input_args=$(printf "%s %s" "args: Using \"\$@\" for ${script_name} arguments:" "$*")
fi

for arg do
  shift
  case "$arg" in
    (--help)
      help=1
      ;;
    (--force)
      force=1
      ;;
    (--config=*)
      config=1
      config_file="$(expr "$arg" : '--config=\(.*\)')"
      ;;
    (*)
      set -- "$@" "$arg"
      ;;
  esac
  # echo "\$@ = " "$@"
done

# Save a copy of the arguments array ($@) to be passed to the env/vars
# scripts. This copy excludes the arguments consumed by this script.
SETVARS_ARGS="$*" ; export SETVARS_ARGS

if [ "$help" != "0" ] ; then
  usage
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 254 ; return
fi

if [ "${SETVARS_COMPLETED:-}" = "1" ] ; then
  if [ $force -eq 0 ] ; then
    echo " "
    echo ":: WARNING: ${script_name} has already been run. Skipping re-execution."
    echo "   To force a re-execution of ${script_name}, use the '--force' option."
    echo "   Using '--force' can result in excessive use of your environment variables."
    usage
    # warning_tally=$(( warning_tally + 1 ))
    # Restore original $@ array before return.
    eval set -- "$script_args" || true
    prep_for_exit 3 ; return
  fi
fi

# If a config file has been supplied, check that it exists and is readable.
if [ "${config:-}" -eq 1 ] ; then
  # fix problem "~" alias, in case it is part of $config_file pathname
  config_file_fix=$(printf "%s" "$config_file" | sed -e "s:^\~:$HOME:")
  if [ ! -r "$config_file_fix" ] ; then
    echo " "
    echo ":: ERROR: $script_name config file could not be found or is not readable."
    echo "   Confirm that \"${config_file}\" path and filename are valid and readable."
    usage
    # Restore original $@ array before return.
    eval set -- "$script_args" || true
    prep_for_exit 4 ; return
  fi
fi


# ############################################################################

# Find those components in the installation folder that include an
# `env/vars.sh` script. We need to "uniq" that list to remove duplicates,
# which happens when multiple versions of a component are installed
# side-by-side.

version_default="latest"

# 2>/dev/null in case of unreadable files/folders generating an error msg
component_array=$(command ls "${script_root}"/*/*/env/vars.sh 2>/dev/null | awk 'NF')

# convert list of `env/vars.sh` files into a list of oneAPI component folders
temp_array=""
eval set -- "$(prep_for_eval "$component_array")"
for arg do
  arg=$(basename -- "$(dirname -- "$(dirname -- "$(dirname -- "$arg")")")")
  if [ -r "${script_root}/${arg}/${version_default}/env/vars.sh" ] ; then
    temp_array=${temp_array}${arg}$posix_nl
  fi
done
component_array=$temp_array

# eliminate duplicate component names and
# get final count of $component_array elements
component_array=$(printf "%s\n" "$component_array" | LC_ALL=C sort | uniq)
temp_var=$(printf "%s\n" "$component_array" | wc -l)

if [ "$temp_var" -le 0 ] ; then
  echo " "
  echo ":: WARNING: No env folders found: No \"env/vars.sh\" scripts to process."
  echo "   The \"${script_name}\" script expects to be located in the installation folder."
  usage
  # warning_tally=$(( warning_tally + 1 ))
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 0 ; return
fi


# ############################################################################

# At this point, if a config file was provided, it is readable.
# Put contents of $config_file into $config_array, and validate content.
# TODO: condense this section; but only if it is worth the effort.

if [ "$config" = "1" ] ; then

  # get the contents of the $config_file and eliminate blank lines
  config_array=$(awk 'NF' "$config_file_fix")

  # Test $config_file: do the requested component paths exist?
  eval set -- "$(prep_for_eval "$config_array")"
  for arg do
    arg_base=$(expr "$arg" : '\(.*\)=.*')
    arg_verz=$(expr "$arg" : '.*=\(.*\)')
    arg_path=${script_root}/${arg_base}/${arg_verz}/env/vars.sh
    # skip test of "default=*" entry here, do it later
    if [ "default" = "$arg_base" ] ; then
      continue
    # skip test of "*=exclude" entry here, do it later
    elif [ "exclude" = "$arg_verz" ] ; then
      continue
    elif [ ! -r "$arg_path" ] || [ "" = "$arg_base" ] ; then
      echo ":: WARNING: Bad config file entry: Unknown component specified."
      echo "   Confirm that \"$arg\" entry in \"$config_file\" is valid."
      warning_tally=$(( warning_tally + 1 ))
    fi
  done

  # Test $config_file: do the requested component versions exist?
  eval set -- "$(prep_for_eval "$config_array")"
  for arg do
    arg_base=$(expr "$arg" : '\(.*\)=.*')
    arg_verz=$(expr "$arg" : '.*=\(.*\)')
    arg_path=${script_root}/${arg_base}/${arg_verz}/env/vars.sh
    # perform "default=*" test we skipped above
    if [ "default" = "$arg_base" ] && [ "exclude" != "$arg_verz" ]; then
      echo ":: ERROR: Bad config file entry: Invalid \"$arg\" entry."
      echo "   \"default=exclude\" is the only valid \"default=\" statement."
      # Restore original $@ array before return.
      eval set -- "$script_args" || true
      prep_for_exit 7 ; return
    elif [ "default" = "$arg_base" ] && [ "exclude" = "$arg_verz" ]; then
      version_default=$arg_verz
      continue
    # perform "*=exclude" test we skipped above (except "default=exclude")
    elif [ "exclude" = "$arg_verz" ] ; then
      # no need to validate the component name, since this is an exclude
      # "*=exclude" lines are ignored when we call the env/vars.sh scripts
      continue
    elif [ ! -r "$arg_path" ] || [ "" = "$arg_verz" ] ; then
      echo ":: WARNING: Bad config file entry: Unknown version \"$arg_verz\" specified."
      echo "   Confirm that \"$arg\" entry in \"$config_file\" is correct."
      warning_tally=$(( warning_tally + 1 ))
    fi
  done

fi


# ############################################################################

# After completing the previous section we know the final "$version_default"
# value. It defaults to "latest" but could have been changed by the
# $config_file to "exclude" by including a "default=exclude" statement.

# add $version_default to all $component_array elements
eval set -- "$(prep_for_eval "$component_array")"
temp_array=""
for arg do
  arg=${arg}"="${version_default}
  temp_array=${temp_array}${arg}$posix_nl
done
component_array=$temp_array


# ############################################################################

# If a config file was provided, add it to the end of our $component_array,
# but only after first removing from the $component_array those that are in
# the $config_array, so we do not initialize a component twice.

if [ "$config" = "1" ] ; then

  # remove components from $component_array that are in $config_array
  eval set -- "$(prep_for_eval "$config_array")"
  for arg do
    arg_base=$(expr "$arg" : '\(.*\)=.*')
    component_array=$(printf "%s\n" "$component_array" | sed -e "s/^$arg_base=.*$//")
  done

  # append $config_array to $component_array to address what we removed
  component_array=${component_array}${posix_nl}${config_array}${posix_nl}

fi

# remove any blank lines resulting from all prior operations
component_array=$(printf "%s\n" "$component_array" | awk 'NF')


# ############################################################################

# Finally! It's time to actually use the $component_array list that we have
# assembled to initialize the oneAPI environment. Up to this point, we've
# minimized any permanent changes to the user's environment, in case of errors
# that might cause a premature exit from this script.

# Adding the SETVARS_CALL=1 parameter to the source arguments list because
# passing arguments to sourced scripts is dicey and inconsistent. Technically,
# the defined positional parameters are supposed to flow automatically through
# to the next sourced shell, but that behavior is inconsistent from shell to
# shell and between macOS and Linux.
# see https://unix.stackexchange.com/questions/441515/parameters-passed-to-a-sourced-script-are-wrong

echo " "
echo ":: initializing oneAPI environment ..."
echo "   $sourcer"
echo "   $input_args"

# ONEAPI_ROOT is expected to point to the top level oneAPI install folder.
# SETVARS_CALL tells env/vars scripts they are being sourced by this script.
ONEAPI_ROOT="${script_root}" ; export ONEAPI_ROOT
SETVARS_CALL=1 ; export SETVARS_CALL

# source the list of components in the $component_array
temp_var=0
eval set -- "$(prep_for_eval "$component_array")"
for arg do
  arg_base=$(expr "$arg" : '\(.*\)=.*')
  arg_verz=$(expr "$arg" : '.*=\(.*\)')
  arg_path=${script_root}/${arg_base}/${arg_verz}/env/vars.sh
  SETVARS_VARS_PATH="${arg_path}" ; export SETVARS_VARS_PATH

  # echo ":: $arg_path"

  if [ "exclude" = "$arg_verz" ] ; then
    continue
  else
    if [ -r "$arg_path" ]; then
      echo ":: $arg_base -- $arg_verz"
      # shellcheck disable=SC2086,SC2240
      # `. script` does not always pass arguments on the command-line, esp. sh/dash
      # passing args as a temporary convenience for updating vars.sh scripts
      . "$arg_path" SETVARS_CALL=1 ${SETVARS_ARGS:-}
      temp_var=$(( temp_var + 1 ))
    else
      echo ":: WARNING: \"$arg_path\" could not be found or is not readable."
      echo "   Confirm that \"$arg_path\" exists and is readable."
      echo "   Could be caused by an incomplete or corrupted product installation."
      warning_tally=$(( warning_tally + 1 ))
    fi
  fi
done

if [ "$temp_var" -eq 0 ] ; then
  echo " "
  echo ":: WARNING: No env scripts found: No \"env/vars.sh\" scripts to process."
  echo "   This can be caused by a bad or incomplete \"--config\" file."
  echo "   Can also be caused by an incomplete or missing oneAPI installation."
  echo "   Can also be caused by redefining 'cd' via an alias or function."
  usage
  # warning_tally=$(( warning_tally + 1 ))
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 0 ; return
fi

echo ":: oneAPI environment initialized ::"
echo " "

if [ "$warning_tally" -ne 0 ] ; then
  echo ":: $warning_tally warnings issued: review warning messages."
  echo "   Possible incomplete environment initialization."
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 250 ; return
else
  # Restore original $@ array before return.
  eval set -- "$script_args" || true
  prep_for_exit 0 ; return
fi
