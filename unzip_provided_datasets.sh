#!/usr/bin/env bash
#
# restore_archive.sh
#
# Restore an object (directory / zarr store) from multi-part ZIP files
#
# Usage:
#   ./restore_archive.sh <zip_directory> <target_directory>
#
set -euo pipefail

# ── colours (disabled if not a terminal) ─────────────────────────────────
if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

# ── helpers ──────────────────────────────────────────────────────────────
info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
die()   { error "$*"; exit 1; }

usage() {
    cat <<EOF
${BOLD}Usage:${NC}
    $(basename "$0") <zip_directory> <target_directory>

${BOLD}Arguments:${NC}
    zip_directory       Directory containing the .zip partial files of the object to be restored
    target_directory    Destination directory for the restored object

${BOLD}Example:${NC}
    $(basename "$0") ./wg_archive ./restored
EOF
    exit 1
}

# ── argument validation ──────────────────────────────────────────────────
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && usage
[ $# -ne 2 ] && { error "Expected exactly 2 arguments, got $#."; usage; }

ZIP_DIR="$1"
TARGET_DIR="$2"

# Resolve to absolute paths for clarity in messages
ZIP_DIR="$(cd "$ZIP_DIR" 2>/dev/null && pwd)" \
    || die "Input directory does not exist or is not accessible: $1"

# ── check that input directory contains zip files ────────────────────────
shopt -s nullglob
ZIP_FILES=("${ZIP_DIR}"/*.zip)
shopt -u nullglob

[ ${#ZIP_FILES[@]} -eq 0 ] && die "No .zip files found in '${ZIP_DIR}'."

info "Found ${#ZIP_FILES[@]} ZIP part(s) in '${ZIP_DIR}'."

# ── check for unzip ─────────────────────────────────────────────────────
command -v unzip >/dev/null 2>&1 \
    || die "'unzip' is not installed. Please install it (e.g. apt install unzip / brew install unzip)."

# ── verify expected part count from manifests (if available) ─────────────
# Peek into the first zip for a manifest to learn the expected total_parts.
MANIFEST_LINE=$(unzip -l "${ZIP_FILES[0]}" 2>/dev/null \
    | grep -oE '_manifest_part_[0-9]+\.json' | head -n1 || true)

if [ -n "${MANIFEST_LINE}" ]; then
    MANIFEST_JSON=$(unzip -p "${ZIP_FILES[0]}" "${MANIFEST_LINE}" 2>/dev/null || true)
    if [ -n "${MANIFEST_JSON}" ]; then
        # Parse total_parts with lightweight tools (grep/sed); no jq dependency
        EXPECTED=$(echo "${MANIFEST_JSON}" \
            | grep -o '"total_parts"[[:space:]]*:[[:space:]]*[0-9]*' \
            | grep -o '[0-9]*$' || true)

        if [ -n "${EXPECTED}" ]; then
            if [ "${#ZIP_FILES[@]}" -ne "${EXPECTED}" ]; then
                warn "Manifest says ${EXPECTED} parts expected, but found ${#ZIP_FILES[@]} ZIP files."
                printf "    Continue anyway? [y/N] "
                read -r REPLY
                [[ "${REPLY}" =~ ^[Yy]$ ]] || die "Aborted by user."
            else
                info "Part count matches manifest (${EXPECTED} parts)."
            fi
        fi
    fi
fi

# ── prepare target directory ─────────────────────────────────────────────
if [ -d "${TARGET_DIR}" ]; then
    # Check if target is non-empty
    if [ "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
        warn "Target directory '${TARGET_DIR}' already exists and is not empty."
        printf "    Existing files may be overwritten. Continue? [y/N] "
        read -r REPLY
        [[ "${REPLY}" =~ ^[Yy]$ ]] || die "Aborted by user."
    fi
else
    info "Creating target directory '${TARGET_DIR}'."
    mkdir -p "${TARGET_DIR}"
fi

# ── extract ──────────────────────────────────────────────────────────────
FAIL_COUNT=0
for zipfile in "${ZIP_FILES[@]}"; do
    BASENAME="$(basename "${zipfile}")"
    info "Extracting ${BASENAME} ..."

    if ! unzip -o -q "${zipfile}" -d "${TARGET_DIR}"; then
        error "Failed to extract ${BASENAME}."
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# ── clean up manifest files from the extraction ──────────────────────────
shopt -s nullglob
MANIFESTS=("${TARGET_DIR}"/_manifest_part_*.json)
shopt -u nullglob

if [ ${#MANIFESTS[@]} -gt 0 ]; then
    info "Removing ${#MANIFESTS[@]} manifest file(s) from target (bookkeeping only)."
    rm -f "${MANIFESTS[@]}"
fi

# ── summary ──────────────────────────────────────────────────────────────
echo ""
if [ "${FAIL_COUNT}" -eq 0 ]; then
    info "${BOLD}Restoration complete.${NC}"
    info "Extracted ${#ZIP_FILES[@]} part(s) into '${TARGET_DIR}'."

    # Show the top-level contents for confirmation
    echo ""
    info "Contents of '${TARGET_DIR}':"
    ls -1F "${TARGET_DIR}"
else
    error "${FAIL_COUNT} out of ${#ZIP_FILES[@]} parts failed to extract."
    error "The restored object may be incomplete."
    exit 1
fi
