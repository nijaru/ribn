use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

impl ArtifactTensor<'_> {
    fn reader_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.reader().read_to_end(&mut bytes).unwrap();
        bytes
    }
}

static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ribn-safetensors-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self) -> PathBuf {
        self.0.join("model.safetensors")
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> Vec<u8> {
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let mut data = Vec::new();
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let mut header = format!(
        r#"{{"weight":{{"dtype":"F32","shape":[2,2],"data_offsets":[0,{}]}}}}"#,
        data.len()
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut artifact = Vec::new();
    artifact.extend_from_slice(
        &u64::try_from(header.len())
            .expect("fixture header length")
            .to_le_bytes(),
    );
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&data);
    artifact
}

/// Several tensors with different shapes, so index offsets are exercised
/// beyond a single leading payload.
fn multi_tensor_fixture() -> Vec<u8> {
    let payloads: [(&str, &[usize], Vec<f32>); 3] = [
        ("first.weight", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ("second.weight", &[3], vec![5.0, 6.0, 7.0]),
        ("third.weight", &[1, 1], vec![8.0]),
    ];
    let mut data = Vec::new();
    let mut header = String::from("{");
    for (position, (name, shape, values)) in payloads.iter().enumerate() {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        if position > 0 {
            header.push(',');
        }
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let _ = write!(
            header,
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{}]}}"#,
            data.len()
        );
    }
    header.push('}');
    let mut header = header.into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut artifact = Vec::new();
    artifact.extend_from_slice(
        &u64::try_from(header.len())
            .expect("fixture header length")
            .to_le_bytes(),
    );
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&data);
    artifact
}

#[test]
fn exposes_format_metadata_and_streamed_payload() {
    let artifact = SafeTensorArtifact::from_bytes(fixture()).expect("artifact");
    assert_eq!(artifact.names(), vec!["weight"]);
    let tensor = artifact.tensor("weight").expect("weight");
    assert_eq!(tensor.scalar_type(), &ScalarType::F32);
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.len(), 16);
    assert!(matches!(
        artifact.tensor("missing"),
        Err(ArtifactError::MissingTensor(name)) if name == "missing"
    ));
}

/// Every tensor keeps its own payload after the metadata is indexed once, so a
/// wrong retained offset cannot pass as a lookup miss.
#[test]
fn retained_offsets_address_each_tensor_payload() {
    let artifact = SafeTensorArtifact::from_bytes(multi_tensor_fixture()).expect("artifact");
    assert_eq!(
        artifact.names(),
        vec!["first.weight", "second.weight", "third.weight"]
    );
    assert_eq!(artifact.tensor_count(), 3);
    assert_eq!(artifact.resident_payload_bytes(), artifact.len());

    let read = |name: &str| {
        artifact
            .tensor(name)
            .expect("tensor")
            .reader_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>()
    };
    assert_eq!(read("first.weight"), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read("second.weight"), vec![5.0, 6.0, 7.0]);
    assert_eq!(read("third.weight"), vec![8.0]);
    assert_eq!(
        artifact.tensor("second.weight").expect("shape").shape(),
        [3]
    );
}

fn encoded(header: &str, payload: usize) -> Vec<u8> {
    let mut header = header.as_bytes().to_vec();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + payload, 0);
    bytes
}

#[test]
fn file_and_memory_validation_match_upstream_for_adversarial_layouts() {
    let dir = TestDir::new();
    let mut cases = vec![
        (
            encoded(
                r#"{"s":{"dtype":"BF16","shape":[],"data_offsets":[0,2]}}"#,
                2,
            ),
            true,
        ),
        (
            encoded(
                r#"{"z":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]}}"#,
                0,
            ),
            true,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                8,
            ),
            false,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            false,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#,
                8,
            ),
            false,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                3,
            ),
            false,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                5,
            ),
            false,
        ),
        (
            encoded(
                r#"{"a":{"dtype":"INVALID","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            false,
        ),
        (
            encoded(
                &format!(
                    r#"{{"a":{{"dtype":"F32","shape":[{},2],"data_offsets":[0,4]}}}}"#,
                    usize::MAX
                ),
                4,
            ),
            false,
        ),
        (vec![0; 7], false),
        (vec![0; 8], false),
        (u64::MAX.to_le_bytes().to_vec(), false),
        (100_000_001_u64.to_le_bytes().to_vec(), false),
    ];
    let mut utf8 = encoded("{}", 0);
    utf8[8] = 255;
    cases.push((utf8, false));
    // Invalid UTF-8 is rejected even in fields unrelated to tensor layout.
    let mut ignored_utf8 = encoded(
        r#"{"s":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"ignored":"X"}}"#,
        4,
    );
    let position = ignored_utf8.iter().position(|b| *b == b'X').unwrap();
    ignored_utf8[position] = 255;
    cases.push((ignored_utf8, false));
    for (bytes, valid) in cases {
        assert_eq!(safetensors::SafeTensors::deserialize(&bytes).is_ok(), valid);
        assert_eq!(SafeTensorArtifact::from_bytes(bytes.clone()).is_ok(), valid);
        std::fs::write(dir.file(), bytes).unwrap();
        assert_eq!(SafeTensorArtifact::open(dir.file()).is_ok(), valid);
    }
}

#[test]
fn file_readers_survive_replacement_drop_and_interleaved_threaded_reads() {
    let dir = TestDir::new();
    let bytes = multi_tensor_fixture();
    std::fs::write(dir.file(), &bytes).unwrap();
    let artifact = SafeTensorArtifact::open(dir.file()).unwrap();
    assert_eq!(artifact.resident_payload_bytes(), 0);
    let mut whole = artifact.reader();
    let readers: Vec<_> = artifact
        .names()
        .into_iter()
        .map(|name| {
            let expected = artifact.tensor(&name).unwrap().reader_bytes();
            (artifact.tensor(&name).unwrap().reader(), expected)
        })
        .collect();
    std::fs::rename(dir.file(), dir.0.join("selected.safetensors")).unwrap();
    std::fs::write(dir.file(), vec![0x55; bytes.len()]).unwrap();
    drop(artifact);
    let mut all = Vec::new();
    std::thread::scope(|scope| {
        for (index, (mut reader, expected)) in readers.into_iter().enumerate() {
            scope.spawn(move || {
                let mut actual = Vec::new();
                let mut buffer = vec![0; index * 7 + 1];
                loop {
                    let n = reader.read(&mut buffer).unwrap();
                    if n == 0 {
                        break;
                    }
                    actual.extend_from_slice(&buffer[..n]);
                    std::thread::yield_now();
                }
                assert_eq!(actual, expected);
                assert_eq!(reader.read(&mut [0; 8]).unwrap(), 0);
            });
        }
        whole.read_to_end(&mut all).unwrap();
    });
    assert_eq!(all, bytes);
}

#[test]
fn large_bf16_payload_is_not_resident_and_reads_stay_inside_tensor_bounds() {
    let dir = TestDir::new();
    let size = 64 * 1024 * 1024;
    let header = format!(
        r#"{{"large":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{size}]}}}}"#,
        size / 2
    );
    let bytes = encoded(&header, 0);
    let mut file = File::create(dir.file()).unwrap();
    file.write_all(&bytes).unwrap();
    file.set_len(u64::try_from(bytes.len() + size).unwrap())
        .unwrap();
    file.write_all(&[1, 2, 3, 4]).unwrap();
    drop(file);
    let artifact = SafeTensorArtifact::open(dir.file()).unwrap();
    assert_eq!(artifact.resident_payload_bytes(), 0);
    let tensor = artifact.tensor("large").unwrap();
    assert_eq!(tensor.scalar_type(), &ScalarType::Bf16);
    assert_eq!(tensor.len(), size);
    let mut reader = tensor.reader();
    let mut first = [0; 4];
    reader.read_exact(&mut first).unwrap();
    assert_eq!(first, [1, 2, 3, 4]);
    assert_eq!(
        std::io::copy(&mut reader, &mut std::io::sink()).unwrap(),
        u64::try_from(size - 4).unwrap()
    );
    assert_eq!(reader.read(&mut first).unwrap(), 0);
}

#[test]
fn premature_file_eof_is_an_error_not_successful_short_tensor() {
    let dir = TestDir::new();
    std::fs::write(dir.file(), fixture()).unwrap();
    let artifact = SafeTensorArtifact::open(dir.file()).unwrap();
    let mut reader = artifact.tensor("weight").unwrap().reader();
    // Deliberate violation of the immutable-source precondition for failure injection.
    File::options()
        .write(true)
        .open(dir.file())
        .unwrap()
        .set_len(0)
        .unwrap();
    assert_eq!(
        reader.read(&mut [0; 4]).unwrap_err().kind(),
        std::io::ErrorKind::UnexpectedEof
    );
}

#[test]
fn cloned_readers_keep_the_selected_source_and_independent_current_cursors() {
    let dir = TestDir::new();
    std::fs::write(dir.file(), fixture()).unwrap();
    let expected: Vec<u8> = [1_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    for artifact in [
        SafeTensorArtifact::open(dir.file()).unwrap(),
        SafeTensorArtifact::from_bytes(fixture()).unwrap(),
    ] {
        let mut reader = artifact.tensor("weight").unwrap().reader();
        let mut prefix = [0; 5];
        reader.read_exact(&mut prefix).unwrap();
        assert_eq!(prefix, expected[..5]);
        let mut cloned = reader.clone();
        drop(artifact);
        let mut first = Vec::new();
        cloned.read_to_end(&mut first).unwrap();
        let mut second = Vec::new();
        reader.read_to_end(&mut second).unwrap();
        assert_eq!(first, expected[5..]);
        assert_eq!(second, expected[5..]);
    }
}
