//! Hugging Face Hub 2.2 shared-blob layout v1, not arbitrary hub-root access.
//! <https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/utils/_shared_blobs.py>

use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SharedBlobs(PathBuf);

impl SharedBlobs {
    pub(super) fn discover(root: PathBuf) -> Option<Self> {
        if !fs::symlink_metadata(&root).ok()?.is_dir() {
            return None;
        }
        let marker = root.join(".huggingface-shared-blobs");
        if !fs::symlink_metadata(&marker).ok()?.is_file() {
            return None;
        }
        // Reject extra content without reading an unbounded/untrusted marker.
        let mut bytes = Vec::with_capacity(3);
        File::open(marker)
            .ok()?
            .take(3)
            .read_to_end(&mut bytes)
            .ok()?;
        (bytes == b"1\n").then_some(Self(root))
    }

    pub(super) fn contains(&self, canonical: &Path) -> bool {
        let Ok(relative) = canonical.strip_prefix(&self.0) else {
            return false;
        };
        let mut parts = relative.components();
        let (Some(Component::Normal(prefix)), Some(Component::Normal(hash)), None) =
            (parts.next(), parts.next(), parts.next())
        else {
            return false;
        };
        let (Some(prefix), Some(hash)) = (prefix.to_str(), hash.to_str()) else {
            return false;
        };
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            && prefix == &hash[..2]
            && fs::symlink_metadata(self.0.join(prefix)).is_ok_and(|m| m.is_dir())
            && fs::symlink_metadata(canonical).is_ok_and(|m| m.is_file())
    }
}
