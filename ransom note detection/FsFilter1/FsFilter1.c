#include <fltKernel.h>
#include <dontuse.h>
#include <suppress.h>

PFLT_FILTER gFilterHandle = NULL;


NTSTATUS DriverEntry(
    __in PDRIVER_OBJECT DriverObject,
    __in PUNICODE_STRING RegistryPath
);
NTSTATUS FilterUnload(
    __in FLT_FILTER_UNLOAD_FLAGS Flags
);
FLT_POSTOP_CALLBACK_STATUS
PostWriteCallback(
    __inout PFLT_CALLBACK_DATA Data,
    __in PCFLT_RELATED_OBJECTS FltObjects,
    __in_opt PVOID CompletionContext,
    __in FLT_POST_OPERATION_FLAGS Flags
);



CONST FLT_OPERATION_REGISTRATION Callbacks[] = {
    { IRP_MJ_WRITE, 0, NULL, PostWriteCallback },
    { IRP_MJ_OPERATION_END }
};


CONST FLT_REGISTRATION FilterRegistration = {
    sizeof(FLT_REGISTRATION),  // Size
    FLT_REGISTRATION_VERSION,  // Version
    0,                         // Flags
    NULL,                      // Context Registration
    Callbacks,                 // Operation Callbacks
    FilterUnload,              // FilterUnload
    NULL,                      // InstanceSetup
    NULL,                      // InstanceQueryTeardown
    NULL,                      // InstanceTeardownStart
    NULL,                      // InstanceTeardownComplete
    NULL,                      // GenerateFileName
    NULL,                      // NormalizeNameComponent
    NULL,                      // NormalizeContextCleanup
    NULL                       // Transaction Notification
};

NTSTATUS
DriverEntry(
    __in PDRIVER_OBJECT DriverObject,
    __in PUNICODE_STRING RegistryPath
)
{
    NTSTATUS status;

    UNREFERENCED_PARAMETER(RegistryPath);

    status = FltRegisterFilter(DriverObject, &FilterRegistration, &gFilterHandle);

    if (NT_SUCCESS(status)) {
        status = FltStartFiltering(gFilterHandle);
        if (!NT_SUCCESS(status)) {
            FltUnregisterFilter(gFilterHandle);
        }
    }

    return status;
}

NTSTATUS
FilterUnload(
    __in FLT_FILTER_UNLOAD_FLAGS Flags
)
{
    UNREFERENCED_PARAMETER(Flags);

    FltUnregisterFilter(gFilterHandle);
    return STATUS_SUCCESS;
}
#define MAX_PATH 260
#include <fltKernel.h>
#include <ntstrsafe.h>
#include <string.h>
#include <stdlib.h>

UNICODE_STRING txtExtension = RTL_CONSTANT_STRING(L"txt");
UNICODE_STRING htmlExtension = RTL_CONSTANT_STRING(L"html");
UNICODE_STRING readExtension = RTL_CONSTANT_STRING(L"readme");
UNICODE_STRING htaExtension = RTL_CONSTANT_STRING(L"hta");
UNICODE_STRING htmExtension = RTL_CONSTANT_STRING(L"htm");

typedef struct _FILTER_WORKITEM_CONTEXT {
    PFLT_INSTANCE Instance;
    PFILE_OBJECT FileObject;
    UNICODE_STRING FileName;
    WCHAR FileNameBuffer[MAX_PATH]; 
} FILTER_WORKITEM_CONTEXT, * PFILTER_WORKITEM_CONTEXT;

BOOLEAN SearchUTF8_MultipleMatches(const char* buffer, size_t bufferLength) {
    if (buffer == NULL || bufferLength == 0) {
        return FALSE;
    }

    const char* dataString = "data";
    const char* fileString = "file";
    const char* lockStrings[] = { "lock", "recover", "restore" };
    const char* cryptString = "crypt";
    const char* mailStrings[] = { "http", "mail" };

    size_t maxMatches = 16;
    size_t dataFileCount = 0;
    size_t lockCryptCount = 0;

    const char* dataFilePositions[1024];
    const char* lockCryptPositions[1024];

    const char* searchPos = buffer;
    while ((searchPos = strstr(searchPos, dataString)) != NULL) {
        if (dataFileCount < maxMatches) {
            dataFilePositions[dataFileCount++] = searchPos;
        }
        searchPos += strlen(dataString);
    }

    searchPos = buffer;
    while ((searchPos = strstr(searchPos, fileString)) != NULL) {
        if (dataFileCount < maxMatches) {
            dataFilePositions[dataFileCount++] = searchPos;
        }
        searchPos += strlen(fileString);
    }

    for (size_t i = 0; i < sizeof(lockStrings) / sizeof(lockStrings[0]); i++) {
        searchPos = buffer;
        while ((searchPos = strstr(searchPos, lockStrings[i])) != NULL) {
            if (lockCryptCount < maxMatches) {
                lockCryptPositions[lockCryptCount++] = searchPos;
            }
            searchPos += strlen(lockStrings[i]);
        }
    }

    searchPos = buffer;
    while ((searchPos = strstr(searchPos, cryptString)) != NULL) {
        if (lockCryptCount < maxMatches) {
            lockCryptPositions[lockCryptCount++] = searchPos;
        }
        searchPos += strlen(cryptString);
    }

    BOOLEAN foundmail = FALSE;
    for (size_t i = 0; i < sizeof(mailStrings) / sizeof(mailStrings[0]); i++) {
        if (strstr(buffer, mailStrings[i]) != NULL) {
            foundmail = TRUE;
            break;
        }
    }

    BOOLEAN foundDataOrFile = (dataFileCount > 0);
    BOOLEAN foundLockOrCrypt = (lockCryptCount > 0);

    if (!(foundDataOrFile && foundLockOrCrypt && foundmail)) {
        return FALSE;
    }

    for (size_t i = 0; i < dataFileCount; i++) {
        for (size_t j = 0; j < lockCryptCount; j++) {
            long diff = labs((long)(lockCryptPositions[j] - dataFilePositions[i]));
            if (diff < 20) {
                return TRUE;
            }
        }
    }

    return FALSE;
}

VOID PostWriteWorkerCallback(
    _Inout_ PFLT_GENERIC_WORKITEM FltWorkItem,
    _In_ PFLT_FILTER Filter,
    _In_opt_ PVOID Context
) {
    PFILTER_WORKITEM_CONTEXT context = (PFILTER_WORKITEM_CONTEXT)Context;
    NTSTATUS status;
    PVOID buffer = NULL;
    ULONG bufferSize = 0;
    ULONG bytesRead = 0;
    LARGE_INTEGER byteOffset;
    FILE_STANDARD_INFORMATION fileInfo;


    status = FltQueryInformationFile(
        context->Instance,
        context->FileObject,
        &fileInfo,
        sizeof(FILE_STANDARD_INFORMATION),
        FileStandardInformation,
        NULL
    );

    if (!NT_SUCCESS(status)) {
        goto cleanup;
    }

    if (RtlCompareUnicodeString(&context->FileName, &txtExtension, TRUE) == 0 ||
        RtlCompareUnicodeString(&context->FileName, &readExtension, TRUE) == 0) {
        bufferSize = 5 * 1024;
    }
    else {
        bufferSize = 30 * 1024;
    }

    if (fileInfo.EndOfFile.QuadPart < bufferSize) {
        bufferSize = (ULONG)fileInfo.EndOfFile.QuadPart;
    }

    if (bufferSize == 0) {
        goto cleanup;
    }

    buffer = ExAllocatePoolWithTag(NonPagedPool, bufferSize + 1, 'tTxt');
    if (buffer == NULL) {
        goto cleanup;
    }

    RtlZeroMemory(buffer, bufferSize + 1);

    byteOffset.QuadPart = 0;
    status = FltReadFile(
        context->Instance,
        context->FileObject,
        &byteOffset,
        bufferSize,
        buffer,
        FLTFL_IO_OPERATION_DO_NOT_UPDATE_BYTE_OFFSET,
        &bytesRead,
        NULL,
        NULL
    );

    if (!NT_SUCCESS(status) || bytesRead == 0) {
        goto cleanup;
    }

    if (bytesRead < bufferSize + 1) {
        ((char*)buffer)[bytesRead] = '\0';
    }
    else {
        ((char*)buffer)[bufferSize] = '\0';
    }


    ULONG validBytes = 0;
    for (ULONG i = 0; i < bytesRead; i++) {
        if (((char*)buffer)[i] != 0x00) {
            validBytes++;
        }
    }

    if (validBytes > 0) {
        char* utf8BufferStart = (char*)ExAllocatePoolWithTag(NonPagedPool, validBytes + 1, 'uTxt');
        char* utf8Buffer = utf8BufferStart;
        if (utf8Buffer) {
            ULONG utf8Index = 0;
            for (ULONG i = 0; i < bytesRead; i++) {
                if (((char*)buffer)[i] != 0x00) {
                    utf8Buffer[utf8Index++] = ((char*)buffer)[i];
                }
            }
            utf8Buffer[utf8Index] = '\0';


            if (utf8Index > 3) {
                if (SearchUTF8_MultipleMatches(utf8Buffer, utf8Index)) {
                    DbgPrint("File Name: %wZ\n", &context->FileName);
                    DbgPrint("Found 'lock' or 'crypt' in the processed UTF-8 file.\n");
                }
            }

            ExFreePoolWithTag(utf8BufferStart, 'uTxt');
        }
        else {
            //DbgPrint("ExAllocatePoolWithTag failed for utf8BufferStart\n");
        }
    }
    else {
        //DbgPrint("PostWriteWorkerCallback: No valid bytes found in buffer.\n");
    }

cleanup:
    if (buffer) {
        ExFreePoolWithTag(buffer, 'tTxt');
    }

    ObDereferenceObject(context->FileObject);

    ExFreePoolWithTag(context, 'wrkC');

    FltFreeGenericWorkItem(FltWorkItem);

    //DbgPrint("PostWriteWorkerCallback: Finished processing.\n");
}

FLT_POSTOP_CALLBACK_STATUS PostWriteCallback(
    __inout PFLT_CALLBACK_DATA Data,
    __in PCFLT_RELATED_OBJECTS FltObjects,
    __in_opt PVOID CompletionContext,
    __in FLT_POST_OPERATION_FLAGS Flags
) {
    NTSTATUS status;
    PFLT_FILE_NAME_INFORMATION fileNameInfo = NULL;
    UNICODE_STRING extension;
    FILE_STANDARD_INFORMATION fileInfo;
    BOOLEAN isLargeTextFile = FALSE;
    BOOLEAN isLargeWebFile = FALSE;

    UNREFERENCED_PARAMETER(CompletionContext);
    UNREFERENCED_PARAMETER(FltObjects);
    UNREFERENCED_PARAMETER(Flags);

    if (FlagOn(Data->Iopb->IrpFlags, IRP_PAGING_IO)) {
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    status = FltGetFileNameInformation(Data, FLT_FILE_NAME_NORMALIZED | FLT_FILE_NAME_QUERY_DEFAULT, &fileNameInfo);
    if (!NT_SUCCESS(status) || fileNameInfo == NULL) {
        //DbgPrint("FltGetFileNameInformation failed with status 0x%08X\n", status);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    FltParseFileNameInformation(fileNameInfo);
    extension = fileNameInfo->Extension;

    status = FltQueryInformationFile(
        FltObjects->Instance,
        FltObjects->FileObject,
        &fileInfo,
        sizeof(FILE_STANDARD_INFORMATION),
        FileStandardInformation,
        NULL
    );

    if (!NT_SUCCESS(status)) {
        //DbgPrint("FltQueryInformationFile failed with status 0x%08X\n", status);
        FltReleaseFileNameInformation(fileNameInfo);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    if (RtlCompareUnicodeString(&extension, &txtExtension, TRUE) == 0 ||
        RtlCompareUnicodeString(&extension, &readExtension, TRUE) == 0) {
        isLargeTextFile = fileInfo.EndOfFile.QuadPart > 5 * 1024;
    }

    if (RtlCompareUnicodeString(&extension, &htmlExtension, TRUE) == 0 ||
        RtlCompareUnicodeString(&extension, &htaExtension, TRUE) == 0 ||
        RtlCompareUnicodeString(&extension, &htmExtension, TRUE) == 0) {
        isLargeWebFile = fileInfo.EndOfFile.QuadPart > 300 * 1024;
    }

    if (isLargeTextFile || isLargeWebFile) {
       // DbgPrint("PostWriteCallback: Large file detected, skipping.\n");
        FltReleaseFileNameInformation(fileNameInfo);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    PFILTER_WORKITEM_CONTEXT workItemContext = (PFILTER_WORKITEM_CONTEXT)ExAllocatePoolWithTag(NonPagedPool, sizeof(FILTER_WORKITEM_CONTEXT), 'wrkC');
    if (workItemContext == NULL) {
        //DbgPrint("PostWriteCallback: ExAllocatePoolWithTag failed for workItemContext\n");
        FltReleaseFileNameInformation(fileNameInfo);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    workItemContext->Instance = FltObjects->Instance;
    workItemContext->FileObject = FltObjects->FileObject;


    if (fileNameInfo->Name.Length + sizeof(WCHAR) > sizeof(workItemContext->FileNameBuffer)) {
        //DbgPrint("PostWriteCallback: File name too long, skipping.\n");
        ExFreePoolWithTag(workItemContext, 'wrkC');
        FltReleaseFileNameInformation(fileNameInfo);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    RtlCopyMemory(workItemContext->FileNameBuffer, fileNameInfo->Name.Buffer, fileNameInfo->Name.Length);
    workItemContext->FileNameBuffer[fileNameInfo->Name.Length / sizeof(WCHAR)] = L'\0'; // ȷ���� NULL ��β
    workItemContext->FileName.Buffer = workItemContext->FileNameBuffer;
    workItemContext->FileName.Length = fileNameInfo->Name.Length;
    workItemContext->FileName.MaximumLength = sizeof(workItemContext->FileNameBuffer);

   // DbgPrint("PostWriteCallback: FileName copied: %wZ\n", &workItemContext->FileName);

    ObReferenceObject(FltObjects->FileObject);

    PFLT_GENERIC_WORKITEM genericWorkItem = FltAllocateGenericWorkItem();
    if (genericWorkItem == NULL) {
        //DbgPrint("PostWriteCallback: FltAllocateGenericWorkItem failed\n");
        ObDereferenceObject(FltObjects->FileObject);
        ExFreePoolWithTag(workItemContext, 'wrkC');
        FltReleaseFileNameInformation(fileNameInfo);
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    status = FltQueueGenericWorkItem(
        genericWorkItem,            
        FltObjects->Filter,         
        PostWriteWorkerCallback,    
        DelayedWorkQueue,           
        workItemContext             
    );

    if (!NT_SUCCESS(status)) {
        //DbgPrint("PostWriteCallback: FltQueueGenericWorkItem failed with status 0x%08X\n", status);
        FltFreeGenericWorkItem(genericWorkItem); 
        ObDereferenceObject(FltObjects->FileObject);
        ExFreePoolWithTag(workItemContext, 'wrkC'); 
    }
    else {
       // DbgPrint("PostWriteCallback: Work item queued successfully.\n");
    }

    FltReleaseFileNameInformation(fileNameInfo);
    return FLT_POSTOP_FINISHED_PROCESSING;
}


