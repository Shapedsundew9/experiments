FROM archlinux/archlinux:latest
RUN pacman -Sy
RUN pacman -S pacman-contrib --noconfirm --disable-download-timeout
RUN curl -s "https://archlinux.org/mirrorlist/?country=DE&protocol=https&use_mirror_status=on" | sed -e 's/^#Server/Server/' -e '/^#/d' | \
    rankmirrors -n 5 - > /etc/pacman.d/mirrorlist

RUN pacman -Syu --noconfirm --disable-download-timeout
RUN pacman -S binutils make gcc fakeroot expac yajl git sudo grep file pyenv --noconfirm --needed
RUN pyenv install 3.11.1 && pyenv global 3.11.1
ENV PATH="/root/.pyenv/shims:$PATH"

RUN pacman -S sudo cgal cairomm sparsehash cairomm python-cairo autoconf-archive autoconf automake pkg-config --noconfirm --needed

RUN apt update -y && apt install -y autotools-dev automake libcgal-dev libboost-all-dev libsparsehash-dev libgtk-3-dev libcairomm-1.0-dev \
    libcairo2-dev pkg-config python3.11-dev python3-matplotlib wget
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz && tar -xvf boost_1_81_0.tar.gz \
    && cd boost_1_81_0 && ./bootstrap.sh --prefix=/usr/ --with-python=python3.11 && CPLUS_INCLUDE_PATH=/usr/include/python3.11 ./b2 install

RUN pip install pycairo numpy scipy matplotlib

ENV MAKEPKG_USER=mkpkg MAKEPKG_GROUP=mkpkg MAKEPKG_ROOT=/tmp/build
RUN groupadd "${MAKEPKG_USER}" && useradd -g "${MAKEPKG_GROUP}" "${MAKEPKG_USER}"
RUN mkdir -p ${MAKEPKG_ROOT}; chown mkpkg:mkpkg ${MAKEPKG_ROOT}
WORKDIR ${MAKEPKG_ROOT}
USER ${MAKEPKG_USER}
ADD PKGBUILD PKGBUILD
RUN makepkg PKGBUILD --needed GIT_REF=master CXXFLAGS="-mtune=generic -O3 -pipe -flto=2 -ffunction-sections -fdata-sections" LDFLAGS="-Wl,--gc-sections"
USER root
RUN mv -v "${MAKEPKG_ROOT}/"python-graph-tool-*-x86_64.pkg.tar.zst /tmp/python-graph-tool.pkg.tar.zst

