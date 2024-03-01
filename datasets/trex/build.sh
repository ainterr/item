#!/usr/bin/env bash

directory="binutils-2.30"
#directory="coreutils-8.29"
#directory="curl-7.71.1"
#directory="diffutils-3.1"
#directory="findutils-4.7.0"
#directory="gmp-6.2.0"
#directory="ImageMagick-7.0.10-27"
#directory="libmicrohttpd-0.9.71"
#directory="libtomcrypt-1.18.2"
#directory="openssl-1.0.1f"
#directory="openssl-1.0.1u"
#directory="openssl-1.0.2h"
#directory="putty-0.74"
#directory="sqlite-3.34.0"
#directory="zlib-1.2.11"


opts=(O0 O1 O2 O3)
archs=(arm-64 mips-64 x86-64)

for opt in ${opts[*]}; do
  for arch in ${archs[*]}; do
    if [[ $arch == arm-64 ]]; then
      prefix=aarch64-linux-gnu-
    elif [[ $arch == mips-64 ]]; then
      prefix=mips64-linux-gnuabi64-
    elif [[ $arch == x86-64 ]]; then
      prefix=
    else
      echo "unsupported arch: $arch"
      exit
    fi

    mkdir -p bin/"$arch"/"$directory"-"$opt"

    tar -xf "$directory".tar.gz
    cd "$directory"

    # --------------------------------------------------------------------------------
    # binutils
    # --------------------------------------------------------------------------------
    if [ -z "$prefix" ]; then
      ./configure CC=${prefix}gcc CFLAGS="-$opt"
    else
      ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    fi

    make

    if [[ -f "./binutils/ar" ]]; then
      cp $(find ./binutils/ -executable -exec file {} \; | grep ELF | cut -d: -f1) ../bin/"$arch"/"$directory"-"$opt"/
    else
      echo "failed to build $arch $directory-$opt"
      exit
    fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # coreutils
    # --------------------------------------------------------------------------------
    #sed -i -e "s/_IO_ftrylockfile/_IO_EOF_SEEN/g" $(find . -type f)
    #sed -i -e '1s/^/#if !defined _IO_IN_BACKUP \&\& defined _IO_EOF_SEEN\n# define _IO_IN_BACKUP 0x100\n#endif\n/' lib/stdio-impl.h

    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./src/ls" ]]; then
    #  cp $(find ./src/ -executable -exec file {} \; | grep ELF | cut -d: -f1) ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # curl
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --without-zlib
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1} --without-zlib
    #fi

    #make

    #if [[ -f "./lib/.libs/libcurl.so.4.6.0" ]]; then
    #  cp ./lib/.libs/libcurl.so.4.6.0 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # diffutils
    # --------------------------------------------------------------------------------
    #sed -i -e "358 s/armv\*-\*/armv*-* | aarch64/" build-aux/config.sub

    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make
    #
    ## ignore security issues in generated code and finish build
    #sed -i -e "1012 s/^/\/\//" lib/stdio.h
    #make

    #if [[ -f "./src/diff" ]]; then
    #  cp $(find ./src/ -executable -exec file {} \; | grep ELF | cut -d: -f1) ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # findutils
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./find/find" ]]; then
    #  cp ./find/find ../bin/"$arch"/"$directory"-"$opt"/
    #  cp ./locate/locate ../bin/"$arch"/"$directory"-"$opt"/
    #  cp ./xargs/xargs ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # gmp
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./.libs/libgmp.so.10.4.0" ]]; then
    #  cp ./.libs/libgmp.so.10.4.0 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # imagemagic
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CXX=${prefix}g++ CFLAGS="-$opt" CXXFLAGS="-$opt" --without-png --without-fontconfig --without-freetype --without-zlib
    #else
    #  ./configure CC=${prefix}gcc CXX=${prefix}g++ CFLAGS="-$opt" CXXFLAGS="-$opt" --host=${prefix::-1} --without-png --without-fontconfig --without-freetype --without-zlib
    #fi

    #make

    #if [[ -f "./Magick++/lib/.libs/libMagick++-7.Q16HDRI.so.4.0.0" ]]; then
    #  cp ./Magick++/lib/.libs/libMagick++-7.Q16HDRI.so.4.0.0 ../bin/"$arch"/"$directory"-"$opt"/
    #  cp ./MagickCore/.libs/libMagickCore-7.Q16HDRI.so.7.0.0 ../bin/"$arch"/"$directory"-"$opt"/
    #  cp ./MagickWand/.libs/libMagickWand-7.Q16HDRI.so.7.0.0 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # libmicrohttpd
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./src/microhttpd/.libs/libmicrohttpd.so.12.56.0" ]]; then
    #  cp ./src/microhttpd/.libs/libmicrohttpd.so.12.56.0 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # libtomcrypt
    # --------------------------------------------------------------------------------
    #make CC=${prefix}gcc CFLAGS="-$opt -fPIC"

    ## build a shared object from the static library
    #ar -x libtomcrypt.a
    #${prefix}gcc -shared -o libtomcrypt.so *.o

    #if [[ -f "./libtomcrypt.so" ]]; then
    #  cp ./libtomcrypt.so ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # openssl
    # --------------------------------------------------------------------------------
    #sed -i "s/-O3 //" ./Configure

    #./Configure linux-elf no-asm -shared -"$opt" --cross-compile-prefix="$prefix"

    #make depend
    #make

    #if [[ -f "./apps/openssl" ]]; then
    #  cp ./apps/openssl ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # putty
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./plink" ]]; then
    #  cp ./plink ./pscp ./psftp ./pterm ./puttytel ./puttygen ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # sqlite
    # --------------------------------------------------------------------------------
    #if [ -z "$prefix" ]; then
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt"
    #else
    #  ./configure CC=${prefix}gcc CFLAGS="-$opt" --host=${prefix::-1}
    #fi

    #make

    #if [[ -f "./sqlite3" ]]; then
    #  cp ./sqlite3 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # zlib
    # --------------------------------------------------------------------------------
    #CC=${prefix}gcc CFLAGS="-$opt" ./configure

    #make

    #if [[ -f "./libz.so.1.2.11" ]]; then
    #  cp ./libz.so.1.2.11 ../bin/"$arch"/"$directory"-"$opt"/
    #else
    #  echo "failed to build $arch $directory-$opt"
    #  exit
    #fi
    # --------------------------------------------------------------------------------

    cd ..
    rm -rf "$directory"
  done
done
