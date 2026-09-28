"""Интеграция с Windows-проводником для выбора Bruker-экспериментов."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int, c_uint, c_void_p
from ctypes import wintypes

try:
    from comtypes import CLSCTX_INPROC_SERVER, COMMETHOD, COMError, GUID, HRESULT, IUnknown, CoCreateInstance

    SIGDN_FILESYSPATH = 0x80058000
    FOS_PICKFOLDERS = 0x00000020
    FOS_FORCEFILESYSTEM = 0x00000040
    FOS_ALLOWMULTISELECT = 0x00000200

    class _ShellItem(IUnknown):
        _iid_ = GUID("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")
        _methods_ = [
            COMMETHOD([], HRESULT, "BindToHandler", (['in'], c_void_p, "pbc"),
                      (['in'], POINTER(GUID), "bhid"), (['in'], POINTER(GUID), "riid"),
                      (['out'], POINTER(c_void_p), "ppv")),
            COMMETHOD([], HRESULT, "GetParent", (['out'], POINTER(c_void_p), "ppsi")),
            COMMETHOD([], HRESULT, "GetDisplayName", (['in'], c_uint, "sigdnName"),
                      (['out'], POINTER(c_void_p), "ppszName")),
            COMMETHOD([], HRESULT, "GetAttributes", (['in'], c_uint, "sfgaoMask"),
                      (['out'], POINTER(c_uint), "psfgaoAttribs")),
            COMMETHOD([], HRESULT, "Compare", (['in'], c_void_p, "psi"), (['in'], c_uint, "hint"),
                      (['out'], POINTER(c_int), "piOrder")),
        ]

    class _ShellItemArray(IUnknown):
        _iid_ = GUID("{B63EA76D-1F85-456F-A19C-48159EFA858B}")
        _methods_ = [
            COMMETHOD([], HRESULT, "BindToHandler", (['in'], c_void_p, "pbc"),
                      (['in'], POINTER(GUID), "bhid"), (['in'], POINTER(GUID), "riid"),
                      (['out'], POINTER(c_void_p), "ppvOut")),
            COMMETHOD([], HRESULT, "GetPropertyStore", (['in'], c_uint, "flags"),
                      (['in'], POINTER(GUID), "riid"),
                      (['out'], POINTER(c_void_p), "ppv")),
            COMMETHOD([], HRESULT, "GetPropertyDescriptionList", (['in'], POINTER(c_void_p), "keyType"),
                      (['in'], POINTER(GUID), "riid"),
                      (['out'], POINTER(c_void_p), "ppv")),
            COMMETHOD([], HRESULT, "GetAttributes", (['in'], c_uint, "attribFlags"),
                      (['in'], c_uint, "sfgaoMask"), (['out'], POINTER(c_uint), "psfgaoAttribs")),
            COMMETHOD([], HRESULT, "GetCount", (['out'], POINTER(c_uint), "pdwNumItems")),
            COMMETHOD([], HRESULT, "GetItemAt", (['in'], c_uint, "index"),
                      (['out'], POINTER(POINTER(_ShellItem)), "ppsi")),
            COMMETHOD([], HRESULT, "EnumItems", (['out'], POINTER(c_void_p), "ppenumShellItems")),
        ]

    class _FileDialog(IUnknown):
        _iid_ = GUID("{42F85136-DB7E-439C-85F1-E4075D135FC8}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Show", (['in'], wintypes.HWND, "parent")),
            COMMETHOD([], HRESULT, "SetFileTypes", (['in'], c_uint, "cFileTypes"),
                      (['in'], c_void_p, "rgFilterSpec")),
            COMMETHOD([], HRESULT, "SetFileTypeIndex", (['in'], c_uint, "iFileType")),
            COMMETHOD([], HRESULT, "GetFileTypeIndex", (['out'], POINTER(c_uint), "iFileType")),
            COMMETHOD([], HRESULT, "Advise", (['in'], c_void_p, "pfde"), (['out'], POINTER(c_uint), "pdwCookie")),
            COMMETHOD([], HRESULT, "Unadvise", (['in'], c_uint, "dwCookie")),
            COMMETHOD([], HRESULT, "SetOptions", (['in'], c_uint, "options")),
            COMMETHOD([], HRESULT, "GetOptions", (['out'], POINTER(c_uint), "options")),
            COMMETHOD([], HRESULT, "SetDefaultFolder", (['in'], POINTER(_ShellItem), "psi")),
            COMMETHOD([], HRESULT, "SetFolder", (['in'], POINTER(_ShellItem), "psi")),
            COMMETHOD([], HRESULT, "GetFolder", (['out'], POINTER(POINTER(_ShellItem)), "ppsi")),
            COMMETHOD([], HRESULT, "GetCurrentSelection", (['out'], POINTER(POINTER(_ShellItem)), "ppsi")),
            COMMETHOD([], HRESULT, "SetFileName", (['in'], wintypes.LPCWSTR, "pszName")),
            COMMETHOD([], HRESULT, "GetFileName", (['out'], POINTER(c_void_p), "pszName")),
            COMMETHOD([], HRESULT, "SetTitle", (['in'], wintypes.LPCWSTR, "title")),
            COMMETHOD([], HRESULT, "SetOkButtonLabel", (['in'], wintypes.LPCWSTR, "label")),
            COMMETHOD([], HRESULT, "SetFileNameLabel", (['in'], wintypes.LPCWSTR, "pszLabel")),
            COMMETHOD([], HRESULT, "GetResult", (['out'], POINTER(POINTER(_ShellItem)), "ppsi")),
            COMMETHOD([], HRESULT, "AddPlace", (['in'], POINTER(_ShellItem), "psi"), (['in'], c_uint, "fdap")),
            COMMETHOD([], HRESULT, "SetDefaultExtension", (['in'], wintypes.LPCWSTR, "pszDefaultExtension")),
            COMMETHOD([], HRESULT, "Close", (['in'], HRESULT, "hr")),
            COMMETHOD([], HRESULT, "SetClientGuid", (['in'], POINTER(GUID), "guid")),
            COMMETHOD([], HRESULT, "ClearClientData"),
            COMMETHOD([], HRESULT, "SetFilter", (['in'], c_void_p, "pFilter")),
        ]

    class _FileOpenDialog(_FileDialog):
        _iid_ = GUID("{D57C7288-D4AD-4768-BE02-9D969532D960}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetResults", (['out'], POINTER(POINTER(_ShellItemArray)), "ppal")),
            COMMETHOD([], HRESULT, "GetSelectedItems", (['out'], POINTER(POINTER(_ShellItemArray)), "ppsai")),
        ]

    _CLSID_FILE_OPEN_DIALOG = GUID("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
    NATIVE_MULTI_FOLDER_DIALOG = True
except (ImportError, OSError):
    NATIVE_MULTI_FOLDER_DIALOG = False


def choose_windows_folders(parent_hwnd: int) -> tuple[str, ...] | None:
    """Открыть стандартный проводник Windows с выбором нескольких папок.

    ``None`` означает, что COM-диалог недоступен и вызывающий код должен
    использовать обычный Tk fallback; пустой кортеж — пользователь отменил
    выбор в проводнике.
    """
    if not NATIVE_MULTI_FOLDER_DIALOG:
        return None
    try:
        dialog = CoCreateInstance(_CLSID_FILE_OPEN_DIALOG, interface=_FileOpenDialog,
                                  clsctx=CLSCTX_INPROC_SERVER)
        options = int(dialog.GetOptions())
        dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_ALLOWMULTISELECT)
        dialog.SetTitle("Выберите папки Bruker (Ctrl/Shift — несколько)")
        dialog.SetOkButtonLabel("Добавить")
        dialog.Show(parent_hwnd)
        result = dialog.GetResults()
    except COMError:
        return ()
    try:
        folders: list[str] = []
        for index in range(int(result.GetCount())):
            item = result.GetItemAt(index)
            raw_path = item.GetDisplayName(SIGDN_FILESYSPATH)
            address = raw_path.value if hasattr(raw_path, "value") else int(raw_path)
            if address:
                try:
                    folders.append(ctypes.wstring_at(address))
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(c_void_p(address))
        return tuple(folders)
    except (COMError, OSError, ValueError):
        return ()
