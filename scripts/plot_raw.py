# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne-qt-browser",
#     "pyqt6",
# ]
# ///


def main() -> None:
    print("Hello from plot_raw.py!")
    import mne
    mne.viz.set_browser_backend("qt")
    raw = mne.io.read_raw_fif(
        "/Users/scotterik/Downloads/UNC-IBIS/derivatives/pylossless/run-02/UNC_7041_v06_20220923_0906_cleaned_raw.fif"
        )
    raw.plot(theme="light")


if __name__ == "__main__":
    main()
