# nerdctl (full distribution)
- nerdctl: v1.7.5
- containerd: v1.7.14
- runc: v1.1.12
- CNI plugins: v1.4.1
- BuildKit: v0.12.5
- Stargz Snapshotter: v0.15.1
- imgcrypt: v1.1.10
- RootlessKit: v2.0.2
- slirp4netns: v1.2.3
- bypass4netns: v0.4.0
- fuse-overlayfs: v1.13
- containerd-fuse-overlayfs: v1.0.8
- Kubo (IPFS): v0.27.0
- Tini: v0.19.0
- buildg: v0.4.1

## License
- bin/slirp4netns:    [GNU GENERAL PUBLIC LICENSE, Version 2](https://github.com/rootless-containers/slirp4netns/blob/v1.2.3/COPYING)
- bin/fuse-overlayfs: [GNU GENERAL PUBLIC LICENSE, Version 2](https://github.com/containers/fuse-overlayfs/blob/v1.13/COPYING)
- bin/ipfs: [Combination of MIT-only license and dual MIT/Apache-2.0 license](https://github.com/ipfs/kubo/blob/v0.27.0/LICENSE)
- bin/{runc,bypass4netns,bypass4netnsd}: Apache License 2.0, statically linked with libseccomp ([LGPL 2.1](https://github.com/seccomp/libseccomp/blob/main/LICENSE), source code available at https://github.com/seccomp/libseccomp/)
- bin/tini: [MIT License](https://github.com/krallin/tini/blob/v0.19.0/LICENSE)
- Other files: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
